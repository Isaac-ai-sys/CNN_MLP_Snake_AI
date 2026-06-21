try:
    import cupy as np
except Exception:
    import numpy as np

import numpy as onp
from collections import deque

import time

from game.snake_env import VectorizedSnakeEnv


class Train:

    def __init__(
        self,
        neural_net,
        board_size=20,
        num_envs=64,
        rollout_steps=1024
    ):
        self.nn = neural_net
        self.board_size = board_size
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        self.position_library = None

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        gamma=0.99,
        lam=0.95
    ):
        """
        rewards: (T, N)
        values:  (T, N)
        dones:   (T, N)
        """

        T, N = rewards.shape

        advantages = np.zeros_like(rewards)

        gae = np.zeros(N, dtype=np.float32)

        next_values = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):

            delta = (
                rewards[t]
                + gamma * next_values * (1.0 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + gamma
                * lam
                * (1.0 - dones[t])
                * gae
            )

            advantages[t] = gae

            next_values = values[t]

        returns = advantages + values

        return advantages, returns
    
    def clear_position_library(self):
        lib = self.position_library
        count = lib["count"]
        lib["snake_boards"] = lib["snake_boards"][count:]
        lib["heads"] = lib["heads"][count:]
        lib["lengths"] = lib["lengths"][count:]
        lib["directions"] = lib["directions"][count:]
        lib["count"] = 0
    
    def add_to_position_library(self, snapshot):
        if snapshot["count"] == 0:
            return

        lib = self.position_library

        def cat(a, b):
            a = a.get() if hasattr(a, "get") else onp.asarray(a)
            b = b.get() if hasattr(b, "get") else onp.asarray(b)
            return onp.concatenate([a, b], axis=0)

        lib["snake_boards"] = cat(lib["snake_boards"], snapshot["snake_boards"])
        lib["heads"]        = cat(lib["heads"],        snapshot["heads"])
        lib["lengths"]      = cat(lib["lengths"],      snapshot["lengths"])
        lib["directions"]   = cat(lib["directions"],   snapshot["directions"])
        lib["count"] = lib["snake_boards"].shape[0]

    def collect_rollout(
        self,
        env,
        epsilon=0.0
    ):
        if self.position_library is not None:
            n_seeded = self.num_envs
            seeded_idx = np.random.choice(self.num_envs, n_seeded, replace=False)
            env.seed_from_library(seeded_idx, self.position_library)

        T = self.rollout_steps
        N = self.num_envs
        S = self.board_size

        states = np.zeros(
            (T, N, 3, S, S),
            dtype=np.float32
        )

        directions = np.zeros(
            (T, N, 4),
            dtype=np.float32
        )

        lengths = np.zeros(
            (T, N),
            dtype=np.float32
        )

        dx_foods = np.zeros(
            (T, N),
            dtype=np.float32
        )

        dy_foods = np.zeros(
            (T, N),
            dtype=np.float32
        )

        runnings = np.zeros(
            (T, N),
            dtype=np.float32
        )

        actions = np.zeros(
            (T, N),
            dtype=np.int32
        )

        rewards = np.zeros(
            (T, N),
            dtype=np.float32
        )

        dones = np.zeros(
            (T, N),
            dtype=np.float32
        )

        values = np.zeros(
            (T, N),
            dtype=np.float32
        )

        log_probs = np.zeros(
            (T, N),
            dtype=np.float32
        )

        final_lengths = np.zeros(
            (T, N),
            dtype=np.float32
        )
        
        ep_start_indexes = np.zeros(
            (N,),
            dtype=np.int16
        )
        
        for t in range(T):

            (
                board,
                direction,
                length,
                dx_food,
                dy_food,
                running
            ) = env.get_state()

            board = np.asarray(board)
            direction = np.asarray(direction)
            length = np.asarray(length)
            dx_food = np.asarray(dx_food)
            dy_food = np.asarray(dy_food)
            running = np.asarray(running)

            probs, value = self.nn.forward_prop(
                board,
                direction,
                length,
                dx_food,
                dy_food,
                running
            )

            probs = np.asarray(probs)

            # epsilon-greedy exploration
            random_mask = (
                np.random.rand(N) < epsilon
            )

            random_actions = np.random.randint(
                0,
                4,
                size=N
            )

            cumprobs = probs.cumsum(axis=1)
            u = np.random.rand(N, 1).astype(probs.dtype)
            sampled_actions = (cumprobs < u).sum(axis=1).clip(0, 3)

            action = np.where(
                random_mask,
                random_actions,
                sampled_actions
            )

            reward = env.step(action)
            reward = np.asarray(reward)

            selected_probs = probs[
                np.arange(N),
                action
            ]

            lp = np.log(
                selected_probs + 1e-8
            )

            done = np.asarray(
                env.running,
                dtype=np.float32
            )
            done = 1.0 - done

            # reset finished environments so new episodes can continue
            dead_envs = np.where(done == 1.0)[0]
            if hasattr(dead_envs, "get"):
                dead_envs = dead_envs.get()
            if dead_envs.size > 0:
                for i, env_idx in enumerate(dead_envs):
                    start = ep_start_indexes[env_idx]
                    final_lengths[start:t+1, env_idx] = env.lengths[env_idx]
                ep_start_indexes[dead_envs] = t + 1  # next episode starts after current t
                env.reset(dead_envs)
                if self.position_library is not None:
                    env.seed_from_library(dead_envs, self.position_library)

            states[t] = board
            directions[t] = direction
            lengths[t] = length

            dx_foods[t] = dx_food
            dy_foods[t] = dy_food

            runnings[t] = running

            actions[t] = action

            rewards[t] = reward

            dones[t] = done

            values[t] = value.squeeze()

            log_probs[t] = lp
        for env_idx in range(N):
            start = ep_start_indexes[env_idx]
            final_lengths[start:, env_idx] = env.lengths[env_idx]  # fill remaining with current length
        return {
            "states": states,
            "directions": directions,
            "lengths": lengths,
            "dx_foods": dx_foods,
            "dy_foods": dy_foods,
            "runnings": runnings,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
            "final_lengths": final_lengths,
        }

    def train(
        self,
        epochs=100,
        actor_learning_rate=0.00015,
        critic_learning_rate=0.0005,
        value_learning_rate=0.0005,
        gamma=0.99,
        lam=0.95,
        ppo_clip=0.2,
        gradient_epochs=4,
        batch_size=16184,
        entropy_coef=0.005,
        value_loss_coef=0.5,
        epsilon=0.0,
        epsilon_decay=0.99,
        epsilon_min=0.00,
        target_kl=0.05,
        verbose=False
    ):

        env = VectorizedSnakeEnv(
            num_envs=self.num_envs,
            size=self.board_size
        )

        for epoch in range(epochs):

            rollout_start = time.perf_counter()

            rollout = self.collect_rollout(
                env,
                epsilon=epsilon
            )

            rollout_end = time.perf_counter()

            states = rollout["states"]
            directions = rollout["directions"]
            lengths = rollout["lengths"]

            dx_foods = rollout["dx_foods"]
            dy_foods = rollout["dy_foods"]

            runnings = rollout["runnings"]

            actions = rollout["actions"]

            rewards = rollout["rewards"]

            dones = rollout["dones"]

            values = rollout["values"]

            old_log_probs = rollout["log_probs"]
            
            final_lengths = rollout["final_lengths"]

            advantages, returns = self.compute_gae(
                rewards,
                values,
                dones,
                gamma=gamma,
                lam=lam
            )

            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std() + 1e-8
            )

            T = self.rollout_steps
            N = self.num_envs

            B = T * N

            states = states.reshape(
                B,
                3,
                self.board_size,
                self.board_size
            )

            directions = directions.reshape(
                B,
                4
            )

            lengths = lengths.reshape(B)

            dx_foods = dx_foods.reshape(B)

            dy_foods = dy_foods.reshape(B)

            runnings = runnings.reshape(B)

            actions = actions.reshape(B)

            returns = returns.reshape(B)

            advantages = advantages.reshape(B)

            old_log_probs = old_log_probs.reshape(B)
            
            final_lengths = final_lengths.reshape(B)

            train_start = time.perf_counter()
            
            entropy_sum = 0.0
            entropy_count = 0

            for g in range(gradient_epochs):

                indices = np.random.permutation(B)

                for start in range(
                    0,
                    B,
                    batch_size
                ):

                    end = start + batch_size

                    batch_idx = indices[start:end]

                    s = states[batch_idx]

                    d = directions[batch_idx]

                    l = lengths[batch_idx]

                    dx = dx_foods[batch_idx]

                    dy = dy_foods[batch_idx]

                    r = runnings[batch_idx]

                    a = actions[batch_idx]

                    ret = returns[batch_idx]

                    adv = advantages[batch_idx]

                    old_lp = old_log_probs[batch_idx]

                    probs, new_values, estimated_lengths = (
                        self.nn.forward_prop_full(
                            s,
                            d,
                            l,
                            dx,
                            dy,
                            r
                        )
                    )
                    
                    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
                    entropy_sum += float(entropy.mean()) * len(a)
                    entropy_count += len(a)

                    selected_probs = probs[
                        np.arange(len(a)),
                        a
                    ]

                    new_log_probs = np.log(
                        selected_probs + 1e-8
                    )

                    ratio = np.exp(
                        new_log_probs - old_lp
                    )

                    clipped_ratio = np.clip(
                        ratio,
                        1.0 - ppo_clip,
                        1.0 + ppo_clip
                    )

                    surrogate = np.minimum(
                        ratio * adv,
                        clipped_ratio * adv
                    )

                    actor_signal = surrogate

                    critic_signal = ret[:, None]

                    entropy = -np.sum(
                        probs * np.log(probs + 1e-8),
                        axis=1
                    )

                    approx_kl = np.mean(
                        old_lp - new_log_probs
                    )

                    if approx_kl > target_kl and g > 0:
                        if verbose:
                            print(
                                f"Early stopping "
                                f"KL={approx_kl:.6f}"
                            )

                        break
                    
                    if g == 0:
                        fl = final_lengths[batch_idx] / (self.board_size * self.board_size)  # normalize
                        self.nn.backward_prop_value_head(targets=fl[:, None], learning_rate=value_learning_rate)

                    self.nn.backward_prop(
                        actions_one_hot=np.eye(4)[a],
                        advantages=actor_signal,
                        returns=critic_signal,
                        actor_learning_rate=actor_learning_rate,
                        critic_learning_rate=critic_learning_rate,
                        entropy_beta=entropy_coef,
                        value_loss_coef=value_loss_coef
                    )
                    

            train_end = time.perf_counter()

            epsilon = max(
                epsilon_min,
                epsilon * epsilon_decay
            )

            avg_returns = returns.mean()

            avg_advantages = advantages.mean()

            avg_length = env.lengths.mean()

            entropy = entropy_sum / entropy_count
            TARGET_ENTROPY = 0.4
            if entropy < TARGET_ENTROPY:
                entropy_coef = min(entropy_coef * 1.005, 0.2)
            elif entropy > 1.1:
                entropy_coef = max(entropy_coef * 0.995, 0.0001)
            # else: leave it alone — entropy is in the healthy zone
            print(
                f"Epoch {epoch} | "
                f"Returns: {avg_returns:.3f} | "
                f"Advantages: {avg_advantages:.3f} | "
                f"Length: {avg_length:.3f} | "
                f"Entropy: {entropy:.3f} | "
                f"Rollout: {rollout_end - rollout_start:.3f}s | "
                f"Train: {train_end - train_start:.3f}s"
            )
        return avg_returns, entropy
    
    def test(self, average_length=40):
        # clear self-play library so it only contains fresh test-run positions
        self.position_library = {
            "snake_boards": onp.zeros((0, self.board_size, self.board_size), dtype=onp.int16),
            "heads": onp.zeros((0, 2), dtype=onp.int32),
            "lengths": onp.zeros((0,), dtype=onp.int32),
            "directions": onp.zeros((0,), dtype=onp.int8),
            "count": 0
        }
        
        env = VectorizedSnakeEnv(
            num_envs=64,
            size=self.board_size
        )
        
        head_history = [(deque(maxlen=self.board_size * 2), set()) for _ in range(self.num_envs)]
        prev_lengths = onp.array(env.lengths.copy())

        max_steps = 10000
        step = 0

        while onp.any(env.running) and step < max_steps:
            step += 1
            # snapshot currently-alive envs before stepping
            alive_idx = onp.where(env.running)[0]
            
            if alive_idx.size > 0:
                # bias snapshotting positions with lengths higher than average_length
                random_num = np.random.randint(10)
                if random_num < 9:
                    long_idx = onp.where(env.running & (env.lengths > average_length))[0]
                    if long_idx.size > 0:
                        snapshot = env.snapshot_envs(long_idx)
                        self.add_to_position_library(snapshot)
                else:
                    snapshot = env.snapshot_envs(alive_idx)
                    self.add_to_position_library(snapshot)

            (
                board,
                direction,
                length,
                dx_food,
                dy_food,
                running
            ) = env.get_state()
            
            #check for loops
            curr_lengths = onp.array(env.lengths)
            for idx in onp.where(env.running)[0]:
                pos = tuple(env.heads[idx])
                dq, seen = head_history[idx]
                
                # if snake ate, reset history — revisiting positions is now valid
                if curr_lengths[idx] > prev_lengths[idx]:
                    dq.clear()
                    seen.clear()
                
                if pos in seen:
                    env.running[idx] = False
                else:
                    if len(dq) == dq.maxlen:
                        seen.discard(dq[0])
                    dq.append(pos)
                    seen.add(pos)

            prev_lengths = curr_lengths

            board = np.asarray(board)
            direction = np.asarray(direction)
            length = np.asarray(length)
            dx_food = np.asarray(dx_food)
            dy_food = np.asarray(dy_food)
            running = np.asarray(running)

            probs, _ = self.nn.forward_prop(
                board,
                direction,
                length,
                dx_food,
                dy_food,
                running
            )

            # argmax on GPU, then bring to CPU for env.step
            actions = np.argmax(probs, axis=1)
            if hasattr(actions, "get"):
                actions = actions.get()

            env.step(actions)

        avg_length = float(onp.mean(env.lengths))
        max_length = int(onp.max(env.lengths))

        print("*** Test ***")
        print(f"  Positions collected: {self.position_library['count']}")
        print(f"  Average Snake Length: {avg_length:.2f}")
        print(f"  Maximum Snake Length: {max_length}")

        return avg_length, max_length