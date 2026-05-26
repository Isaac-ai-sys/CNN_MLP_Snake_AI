try:
    import cupy as np
except Exception:
    import numpy as np

from game.snake_env import Snake_Env
import time

class Train():
    def __init__(self, neural_net, board_size=20):
        self.nn = neural_net
        self.board_size = board_size

    @staticmethod
    def collect_episode(nn, board_size, max_steps, epsilon=0.0):
        states, actions, values, rewards, directions, lengths, distances_to_danger_up, distances_to_danger_right, distances_to_danger_down, distances_to_danger_left, dx_foods, dy_foods, runnings, log_probs = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        env = Snake_Env(board_size)
        step = 0
        while env.running and step < max_steps:
            state, direction, length, distance_to_danger_up, distance_to_danger_right, distance_to_danger_down, distance_to_danger_left, dx_food, dy_food, running = env.get_state()
            states.append(state)
            directions.append(direction)
            lengths.append(length)
            distances_to_danger_up.append(distance_to_danger_up)
            distances_to_danger_right.append(distance_to_danger_right)
            distances_to_danger_down.append(distance_to_danger_down)
            distances_to_danger_left.append(distance_to_danger_left)
            dx_foods.append(dx_food)
            dy_foods.append(dy_food)
            runnings.append(running)
            action, value, log_prob = nn.choose_action(
                state,
                direction,
                length,
                distance_to_danger_up,
                distance_to_danger_right,
                distance_to_danger_down,
                distance_to_danger_left,
                dx_food,
                dy_food,
                running,
                epsilon
            )
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(env.step(np.argmax(action)))
            step += 1
        
        return (states, directions, lengths, actions, values, rewards, log_probs, distances_to_danger_up, distances_to_danger_right, distances_to_danger_down, distances_to_danger_left, dx_foods, dy_foods, runnings), lengths[-1]

    @staticmethod
    def collect_episodes(nn, board_size, max_steps, episodes, epsilon=0.0, num_parallel_envs=4):
        num_parallel_envs = min(num_parallel_envs, episodes)
        envs = [Snake_Env(board_size) for _ in range(num_parallel_envs)]

        buffers = [
            {
                'states': [],
                'directions': [],
                'lengths': [],
                'distances_to_danger_up': [],
                'distances_to_danger_right': [],
                'distances_to_danger_down': [],
                'distances_to_danger_left': [],
                'dx_foods': [],
                'dy_foods': [],
                'runnings': [],
                'actions': [],
                'values': [],
                'log_probs': [],
                'rewards': []
            }
            for _ in range(num_parallel_envs)
        ]

        episode_data = []
        snake_length_sum = 0.0
        next_env_index = 0

        active_indices = list(range(num_parallel_envs))
        while len(episode_data) < episodes:
            batch_states = []
            batch_directions = []
            batch_lengths = []
            batch_du = []
            batch_dr = []
            batch_dd = []
            batch_dl = []
            batch_dx = []
            batch_dy = []
            batch_running = []

            for idx in active_indices:
                env = envs[idx]
                state, direction, length, distance_to_danger_up, distance_to_danger_right, distance_to_danger_down, distance_to_danger_left, dx_food, dy_food, running = env.get_state()
                batch_states.append(state)
                batch_directions.append(direction)
                batch_lengths.append(length)
                batch_du.append(distance_to_danger_up)
                batch_dr.append(distance_to_danger_right)
                batch_dd.append(distance_to_danger_down)
                batch_dl.append(distance_to_danger_left)
                batch_dx.append(dx_food)
                batch_dy.append(dy_food)
                batch_running.append(running)

            actions_one_hot, action_indices, values, log_probs = nn.choose_actions_batch(
                np.asarray(batch_states),
                np.asarray(batch_directions),
                np.asarray(batch_lengths, dtype=np.float32),
                np.asarray(batch_du, dtype=np.float32),
                np.asarray(batch_dr, dtype=np.float32),
                np.asarray(batch_dd, dtype=np.float32),
                np.asarray(batch_dl, dtype=np.float32),
                np.asarray(batch_dx, dtype=np.float32),
                np.asarray(batch_dy, dtype=np.float32),
                np.asarray(batch_running, dtype=np.float32),
                epsilon
            )

            next_active = []
            for batch_idx, idx in enumerate(active_indices):
                env = envs[idx]
                buf = buffers[idx]
                buf['states'].append(batch_states[batch_idx])
                buf['directions'].append(batch_directions[batch_idx])
                buf['lengths'].append(batch_lengths[batch_idx])
                buf['distances_to_danger_up'].append(batch_du[batch_idx])
                buf['distances_to_danger_right'].append(batch_dr[batch_idx])
                buf['distances_to_danger_down'].append(batch_dd[batch_idx])
                buf['distances_to_danger_left'].append(batch_dl[batch_idx])
                buf['dx_foods'].append(batch_dx[batch_idx])
                buf['dy_foods'].append(batch_dy[batch_idx])
                buf['runnings'].append(batch_running[batch_idx])

                action = action_indices[batch_idx]
                if hasattr(action, 'get'):
                    action = int(action.get())
                else:
                    action = int(action)

                buf['actions'].append(np.asarray(actions_one_hot[batch_idx]))
                buf['values'].append(values[batch_idx])
                buf['log_probs'].append(log_probs[batch_idx])

                reward = env.step(action)
                buf['rewards'].append(reward)

                if env.running and len(buf['rewards']) < max_steps:
                    next_active.append(idx)
                else:
                    episode_data.append((
                        (
                            buf['states'],
                            buf['directions'],
                            buf['lengths'],
                            buf['actions'],
                            buf['values'],
                            buf['rewards'],
                            buf['log_probs'],
                            buf['distances_to_danger_up'],
                            buf['distances_to_danger_right'],
                            buf['distances_to_danger_down'],
                            buf['distances_to_danger_left'],
                            buf['dx_foods'],
                            buf['dy_foods'],
                            buf['runnings']
                        ),
                        buf['lengths'][-1]
                    ))
                    snake_length_sum += buf['lengths'][-1]

                    if len(episode_data) < episodes:
                        envs[idx] = Snake_Env(board_size)
                        buffers[idx] = {
                            'states': [],
                            'directions': [],
                            'lengths': [],
                            'distances_to_danger_up': [],
                            'distances_to_danger_right': [],
                            'distances_to_danger_down': [],
                            'distances_to_danger_left': [],
                            'dx_foods': [],
                            'dy_foods': [],
                            'runnings': [],
                            'actions': [],
                            'values': [],
                            'log_probs': [],
                            'rewards': []
                        }
                        next_active.append(idx)

            active_indices = next_active
            if len(active_indices) == 0 and len(episode_data) < episodes:
                active_indices = [i for i in range(num_parallel_envs)]

        return episode_data, snake_length_sum
    
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        rewards = np.array(rewards)
        values = np.array(values).reshape(-1)

        values_next = np.append(values[1:], 0.0)

        deltas = rewards + gamma * values_next - values

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * lam * gae
            advantages[t] = gae

        return advantages
    
    def train(self, epochs=100, episodes=100, max_steps=1000, actor_learning_rate=0.0001, critic_learning_rate=0.0001, value_loss_coef=0.5, entropy_coef=0.1, verbose=False, advantage_scale=2.0, epsilon=0.0, target_kl=0.3, min_actor_lr=1e-6):
        epoch_avg = 0
        entropy_avg = 0
        gradient_updates = 0
        epsilon_end = 0
        epsilon_decay = 0.5
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            size = self.board_size
            # Reset learning rates at the start of each epoch
            actor_learning_rate_epoch = actor_learning_rate
            critic_learning_rate_epoch = critic_learning_rate

            # Episode collection is batched across multiple environments to reduce GPU inference overhead
            start = time.perf_counter()
            episode_data, snake_length_sum = self.collect_episodes(
                self.nn,
                self.board_size,
                max_steps,
                episodes,
                epsilon=epsilon,
                num_parallel_envs=min(16, episodes)
            )
            end = time.perf_counter()

            print(f"Episode Execution time: {end - start:.6f} seconds")

            # average final length per episode
            avg = snake_length_sum / episodes
            epoch_avg += avg
            # print(f"AVG Snake Length for epoch {e}: {avg:.4f}")
            
            all_states = np.stack([np.asarray(state) for ep in episode_data for state in ep[0][0]])
            all_directions = np.stack([np.asarray(direction) for ep in episode_data for direction in ep[0][1]])
            all_lengths = np.asarray([np.asarray(l, dtype=np.float32) for ep in episode_data for l in ep[0][2]], dtype=np.float32)
            all_actions = np.stack([np.asarray(action) for ep in episode_data for action in ep[0][3]])
            all_distances_to_danger_up = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][7]], dtype=np.float32)
            all_distances_to_danger_right = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][8]], dtype=np.float32)
            all_distances_to_danger_down = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][9]], dtype=np.float32)
            all_distances_to_danger_left = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][10]], dtype=np.float32)
            all_dx_foods = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][11]], dtype=np.float32)
            all_dy_foods = np.asarray([np.asarray(d, dtype=np.float32) for ep in episode_data for d in ep[0][12]], dtype=np.float32)
            all_runnings = np.asarray([np.asarray(r, dtype=np.float32) for ep in episode_data for r in ep[0][13]], dtype=np.float32)
            all_advantages = []
            all_returns = []
            episode_rewards = []
            reward_sum = 0
            reward_count = 1
            for idx, ep in enumerate(episode_data):
                rewards = np.array(ep[0][5])
                reward_sum += np.sum(rewards)
                reward_count += 1
                values = np.array(ep[0][4]).squeeze()

                adv = self.compute_gae(rewards, values)
                ret = adv + values

                all_returns.append(ret)
                all_advantages.append(adv)
                episode_rewards.append(rewards.sum())
                if verbose:
                    print(f"Episode {idx}: total_reward={rewards.sum():.3f}, mean_raw_adv={adv.mean():.6f}, std_raw_adv={adv.std():.6f}, steps={len(rewards)}")
            print(f"Raw Reward Averages: {(reward_sum / reward_count):.3f}")
            all_returns = np.concatenate(all_returns)
            # print(f"Raw returns averages: {all_returns.mean():.3f}")
            # concatenate across episodes
            raw_adv_concat = np.concatenate(all_advantages)
            episode_rewards_arr = np.asarray(episode_rewards, dtype=np.float32)
            if verbose:
                print(f"Pre-normalization advantages: mean={raw_adv_concat.mean():.6f}, std={raw_adv_concat.std():.6f}")
                print(f"Episode rewards: mean={episode_rewards_arr.mean():.3f}, std={episode_rewards_arr.std():.3f}")
            all_advantages = (raw_adv_concat - raw_adv_concat.mean()) / (raw_adv_concat.std() + 1e-8)
            # apply advantage scaling to amplify policy gradient signal
            if advantage_scale != 1.0:
                all_advantages = all_advantages * advantage_scale
                if verbose:
                    print(f"Applied advantage_scale={advantage_scale}")

            eps = 0.2
            batch_size = min(len(all_states) // 32, 2056)
            gradient_epochs = 3
            
            # Compute old log probs once from the current (pre-update) policy
            start = time.perf_counter()
            debug_num_states = len(all_states)
            debug_num_batches = (debug_num_states + batch_size - 1) // batch_size
            old_log_probs = []
            for i in range(0, len(all_states), batch_size):
                s = all_states[i:i+batch_size]
                d = all_directions[i:i+batch_size]
                l = all_lengths[i:i+batch_size]
                a = all_actions[i:i+batch_size]
                du = all_distances_to_danger_up[i:i+batch_size]
                dr = all_distances_to_danger_right[i:i+batch_size]
                dd = all_distances_to_danger_down[i:i+batch_size]
                dl = all_distances_to_danger_left[i:i+batch_size]
                dx = all_dx_foods[i:i+batch_size]
                dy = all_dy_foods[i:i+batch_size]
                runs = all_runnings[i:i+batch_size]
                probs, _ = self.nn.forward_prop(s, d, l, du, dr, dd, dl, dx, dy, runs)
                action_indices = np.argmax(a, axis=1)
                lp = np.log(probs[np.arange(len(action_indices)), action_indices] + 1e-8)
                old_log_probs.append(lp)
            old_log_probs = np.concatenate(old_log_probs)
            end_old_log_probs = time.perf_counter()
            debug_old_log_probs_time = end_old_log_probs - start

            start_grad = time.perf_counter()
            early_stopped = False
            for g in range(gradient_epochs):
                if early_stopped:
                    break
                stop_update = False
                idx = np.random.permutation(len(all_states))
                grad_epoch_start = time.perf_counter()
                batch_count = 0
                for i in range(0, len(all_states), batch_size):
                    batch_count += 1
                    gradient_updates += 1
                    indices = idx[i:i+batch_size]
                    s = all_states[indices]
                    d = all_directions[indices]
                    l = all_lengths[indices]
                    a = all_actions[indices]
                    du = all_distances_to_danger_up[indices]
                    dr = all_distances_to_danger_right[indices]
                    dd = all_distances_to_danger_down[indices]
                    dl = all_distances_to_danger_left[indices]
                    dx = all_dx_foods[indices]
                    dy = all_dy_foods[indices]
                    runs = all_runnings[indices]

                    adv = all_advantages[indices]
                    ret = all_returns[indices]

                    new_probs, new_values = self.nn.forward_prop(s, d, l, du, dr, dd, dl, dx, dy, runs)
                    ret = np.asarray(ret, dtype=np.float32)

                    critic_mae = np.mean(np.abs(new_values.squeeze() - ret))

                    entropy = -np.sum(
                        new_probs * np.log(new_probs + 1e-8),
                        axis=1
                    )

                    entropy_avg += entropy.mean()

                    action_indices = np.argmax(a, axis=1)

                    new_log_probs = np.log(
                        new_probs[np.arange(len(action_indices)), action_indices] + 1e-8
                    )

                    # approximate KL(old || new) on this minibatch
                    approx_kl = np.mean(old_log_probs[indices] - new_log_probs)
                    if approx_kl > target_kl:
                        if verbose:
                            print(f"  early stopping: approx_kl={approx_kl:.6f} > target_kl={target_kl:.6f}")
                        # be more conservative for subsequent updates in this epoch
                        actor_learning_rate_epoch = max(actor_learning_rate_epoch * 0.5, min_actor_lr)
                        stop_update = True
                        early_stopped = True

                    ratio = np.exp(new_log_probs - old_log_probs[indices])

                    clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
                    ppo_weight = np.where(
                        adv >= 0,
                        np.minimum(ratio, clipped_ratio),
                        np.maximum(ratio, clipped_ratio)
                    )

                    ppo_weight *= adv

                    critic_loss_signal = ret[:, None]

                    if verbose:
                        # compute diagnostics for this batch
                        mean_ratio = np.mean(ratio)
                        clipped_frac = np.mean((np.abs(ratio - 1.0) > eps).astype(np.float32))
                        mean_adv = np.mean(adv)
                        mean_entropy = np.mean(entropy)
                        mean_action_prob = np.mean(new_probs[np.arange(len(action_indices)), action_indices])
                        print(f"Batch {gradient_updates}: mean_ratio={mean_ratio:.3f}, clipped_frac={clipped_frac:.3f}, mean_adv={mean_adv:.3f}, mean_entropy={mean_entropy:.3f}, mean_action_prob={mean_action_prob:.3f}, critic_mae={critic_mae:.3f}")
                        if clipped_frac > 0:
                            print(f"  warning: clipped_frac={clipped_frac:.3f} indicates some ratio clipping")

                    self.nn.backward_prop(
                        a,
                        ppo_weight,
                        critic_loss_signal,
                        actor_learning_rate_epoch,
                        critic_learning_rate_epoch,
                        entropy_beta=entropy_coef,
                        value_loss_coef=value_loss_coef,
                    )

                    if stop_update:
                        break
                    
                    if verbose:
                        actor_dw = np.asarray([getattr(layer, 'last_dw_norm', 0.0) for layer in self.nn.actor_layers], dtype=np.float32)
                        actor_db = np.asarray([getattr(layer, 'last_db_norm', 0.0) for layer in self.nn.actor_layers], dtype=np.float32)
                        print(f"Actor grad norms: mean_dw={actor_dw.mean():.6f}, mean_db={actor_db.mean():.6f}, per_layer_dw={actor_dw}")

                    # raw_avg = snake_length_sum / episodes
                    # print(f"Epoch {e}: avg_length={raw_avg:.2f}, grad_epochs_run={g+1}")
                grad_epoch_end = time.perf_counter()
                if True:  # Set to True to debug gradient epoch timing
                    print(f"  Grad epoch {g}: {grad_epoch_end - grad_epoch_start:.3f}s ({batch_count} batches)")
            end_grad = time.perf_counter()
            debug_grad_time = end_grad - start_grad
            end = time.perf_counter()
            epsilon = max(
                epsilon_end,
                epsilon * epsilon_decay
            )
            print(f"Backpropogation Execution time: {end - start:.6f} seconds (num_states={debug_num_states}, batches={debug_num_batches}, old_log_probs={debug_old_log_probs_time:.3f}s, grad_epochs={debug_grad_time:.3f}s)")
        epoch_avg *= self.board_size * self.board_size
        epoch_avg /= epochs
        entropy_avg /= max(1, gradient_updates)
        return epoch_avg, entropy_avg