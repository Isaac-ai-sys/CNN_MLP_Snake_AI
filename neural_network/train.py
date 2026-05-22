import numpy as np
from game.snake_env import Snake_Env
import multiprocessing

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
            rewards.append(env.step(action))
            step += 1
        
        return (states, directions, lengths, actions, values, rewards, log_probs, distances_to_danger_up, distances_to_danger_right, distances_to_danger_down, distances_to_danger_left, dx_foods, dy_foods, runnings), lengths[-1]
    
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
    
    def train(self, epochs=100, episodes=100, max_steps=1000, actor_learning_rate=0.0001, critic_learning_rate=0.0001, value_loss_coef=0.5, entropy_coef=0.02, verbose=False, advantage_scale=1.0, epsilon=0.0, target_kl=0.03, min_actor_lr=1e-6):
        epoch_avg = 0
        entropy_avg = 0
        gradient_updates = 0
        epsilon_start = epsilon
        epsilon_end = 0
        epsilon_decay = 0.5
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            size = self.board_size
            
            # Episode collection (use single process for determinism and easier debugging)
            for _ in range(episodes):
                ep_data, final_length = self.collect_episode(self.nn, self.board_size, max_steps, epsilon)
                episode_data.append(ep_data)
                snake_length_sum += final_length

            # average final length per episode
            avg = snake_length_sum / episodes
            epoch_avg += avg
            # print(f"AVG Snake Length for epoch {e}: {avg:.4f}")
            
            all_states = np.concatenate([ep[0] for ep in episode_data])
            all_directions = np.concatenate([ep[1] for ep in episode_data])
            all_lengths = np.array([l for ep in episode_data for l in ep[2]])
            all_actions = np.concatenate([ep[3] for ep in episode_data])
            all_distances_to_danger_up = np.array([d for ep in episode_data for d in ep[7]])
            all_distances_to_danger_right = np.array([d for ep in episode_data for d in ep[8]])
            all_distances_to_danger_down = np.array([d for ep in episode_data for d in ep[9]])
            all_distances_to_danger_left = np.array([d for ep in episode_data for d in ep[10]])
            all_dx_foods = np.array([d for ep in episode_data for d in ep[11]])
            all_dy_foods = np.array([d for ep in episode_data for d in ep[12]])
            all_runnings = np.array([r for ep in episode_data for r in ep[13]])
            all_advantages = []
            all_returns = []
            episode_rewards = []

            for idx, ep in enumerate(episode_data):
                rewards = np.array(ep[5])
                values = np.array(ep[4]).squeeze()

                adv = self.compute_gae(rewards, values)
                ret = adv + values

                all_returns.append(ret)
                all_advantages.append(adv)
                episode_rewards.append(rewards.sum())
                if verbose:
                    print(f"Episode {idx}: total_reward={rewards.sum():.3f}, mean_raw_adv={adv.mean():.6f}, std_raw_adv={adv.std():.6f}, steps={len(rewards)}")

            all_returns = np.concatenate(all_returns)
            # concatenate across episodes
            raw_adv_concat = np.concatenate(all_advantages)
            if verbose:
                print(f"Pre-normalization advantages: mean={raw_adv_concat.mean():.6f}, std={raw_adv_concat.std():.6f}")
                print(f"Episode rewards: mean={np.mean(episode_rewards):.3f}, std={np.std(episode_rewards):.3f}")
            all_advantages = (raw_adv_concat - raw_adv_concat.mean()) / (raw_adv_concat.std() + 1e-8)
            # apply advantage scaling to amplify policy gradient signal
            if advantage_scale != 1.0:
                all_advantages = all_advantages * advantage_scale
                if verbose:
                    print(f"Applied advantage_scale={advantage_scale}")

            eps = 0.2
            batch_size = 64
            gradient_epochs = 1
            
            # Compute old log probs once from the current (pre-update) policy
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
            
            for g in range(gradient_epochs):
                stop_update = False
                idx = np.random.permutation(len(all_states))
                for i in range(0, len(all_states), batch_size):
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
                        actor_learning_rate = max(actor_learning_rate * 0.5, min_actor_lr)
                        stop_update = True

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
                        actor_learning_rate,
                        critic_learning_rate,
                        entropy_beta=entropy_coef,
                        value_loss_coef=value_loss_coef,
                    )

                    if stop_update:
                        break
                    
                    if verbose:
                        actor_dw = [getattr(layer, 'last_dw_norm', 0.0) for layer in self.nn.actor_layers]
                        actor_db = [getattr(layer, 'last_db_norm', 0.0) for layer in self.nn.actor_layers]
                        print(f"Actor grad norms: mean_dw={np.mean(actor_dw):.6f}, mean_db={np.mean(actor_db):.6f}, per_layer_dw={actor_dw}")

                    # raw_avg = snake_length_sum / episodes
                    # print(f"Epoch {e}: avg_length={raw_avg:.2f}, grad_epochs_run={g+1}")
            epsilon = max(
                epsilon_end,
                epsilon_start * epsilon_decay
            )
        epoch_avg *= self.board_size * self.board_size
        epoch_avg /= epochs
        entropy_avg /= max(1, gradient_updates)
        return epoch_avg, entropy_avg