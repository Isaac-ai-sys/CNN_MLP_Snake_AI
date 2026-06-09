try:
    import cupy as np
except Exception:
    import numpy as np

import time

from game.snake_env import VectorizedSnakeEnv


class Train:

    def __init__(
        self,
        neural_net,
        board_size=20,
        num_envs=64,
        rollout_steps=256
    ):
        self.nn = neural_net
        self.board_size = board_size
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

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

    def collect_rollout(
        self,
        env,
        epsilon=0.0
    ):

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

            sampled_actions = np.array([
                np.random.choice(
                    4,
                    size=1,
                    p=probs[i]
                )[0]
                for i in range(N)
            ])

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
                env.reset(dead_envs)

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
        }

    def train(
        self,
        epochs=100,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.0001,
        gamma=0.99,
        lam=0.95,
        ppo_clip=0.2,
        gradient_epochs=4,
        batch_size=4096,
        entropy_coef=0.05,
        value_loss_coef=0.5,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        target_kl=0.03,
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

            train_start = time.perf_counter()

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

                    probs, new_values = (
                        self.nn.forward_prop(
                            s,
                            d,
                            l,
                            dx,
                            dy,
                            r
                        )
                    )
                    
                    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
                    if entropy < 0.5:  # log(4) ≈ 1.386 is max for 4 actions
                        entropy_coef = min(entropy_coef * 1.05, 0.2)  # slightly increase
                    elif entropy > 1:
                        entropy_coef = max(entropy_coef * 0.95, 0.05) # slightly decrease

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

                    if approx_kl > target_kl:

                        if verbose:
                            print(
                                f"Early stopping "
                                f"KL={approx_kl:.6f}"
                            )

                        break

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

            avg_reward = rewards.mean()

            avg_length = env.lengths.mean()

            alive = env.running.mean()

            entropy = float(-np.sum(probs * np.log(probs + 1e-8), axis=1).mean())
            print(
                f"Epoch {epoch} | "
                f"Reward: {avg_reward:.3f} | "
                f"Length: {avg_length:.3f} | "
                f"Entropy: {entropy:.3f} | "
                f"Alive: {alive:.3f} | "
                f"Rollout: {rollout_end - rollout_start:.3f}s | "
                f"Train: {train_end - train_start:.3f}s"
            )
        return avg_length, entropy