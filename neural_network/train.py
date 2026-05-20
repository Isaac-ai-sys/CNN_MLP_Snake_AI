import numpy as np
from game.snake_env import Snake_Env
import multiprocessing

class Train():
    def __init__(self, neural_net, board_size=20):
        self.nn = neural_net
        self.board_size = board_size
    
    @staticmethod
    def collect_episode(nn, board_size, max_steps):
        states, actions, values, rewards, directions, lengths, distances_to_danger_up, distances_to_danger_right, distances_to_danger_down, distances_to_danger_left, dx_foods, dy_foods, log_probs = [], [], [], [], [], [], [], [], [], [], [], [], []
        env = Snake_Env(board_size)
        step = 0
        while env.running and step < max_steps:
            state, direction, length, distance_to_danger_up, distance_to_danger_right, distance_to_danger_down, distance_to_danger_left, dx_food, dy_food = env.get_state()
            states.append(state)
            directions.append(direction)
            lengths.append(length)
            distances_to_danger_up.append(distance_to_danger_up)
            distances_to_danger_right.append(distance_to_danger_right)
            distances_to_danger_down.append(distance_to_danger_down)
            distances_to_danger_left.append(distance_to_danger_left)
            dx_foods.append(dx_food)
            dy_foods.append(dy_food)
            action, value, log_prob = nn.choose_action(state, direction, length, distance_to_danger_up, distance_to_danger_right, distance_to_danger_down, distance_to_danger_left, dx_food, dy_food)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(env.step(action))
            step += 1
        
        return (states, directions, lengths, actions, values, rewards, log_probs, distances_to_danger_up, distances_to_danger_right, distances_to_danger_down, distances_to_danger_left, dx_foods, dy_foods), lengths[-1]
    
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
    
    def train(self, epochs=100, episodes=100, max_steps=1000, learning_rate=0.0001):
        epoch_avg = 0
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            size = self.board_size
            
            # Multiprocess episode collection
            with multiprocessing.Pool() as pool:
                results = pool.starmap(self.collect_episode, [(self.nn, self.board_size, max_steps)] * episodes)
            
            for ep_data, final_length in results:
                episode_data.append(ep_data)
                snake_length_sum += final_length

            avg = (snake_length_sum * size * size) / episodes
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
            all_advantages = []
            all_returns = []

            for ep in episode_data:
                rewards = np.array(ep[5])
                values = np.array(ep[4]).squeeze()

                adv = self.compute_gae(rewards, values)
                ret = adv + values

                all_returns.append(ret)
                all_advantages.append(adv)
            
            all_returns = np.concatenate(all_returns)
            # concatenate across episodes
            all_advantages = np.concatenate(all_advantages)
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            eps = 0.2
            batch_size = 64
            gradient_epochs = 3
            
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
                probs, _ = self.nn.forward_prop(s, d, l, du, dr, dd, dl, dx, dy)
                action_indices = np.argmax(a, axis=1)
                lp = np.log(probs[np.arange(len(action_indices)), action_indices] + 1e-8)
                old_log_probs.append(lp)
            old_log_probs = np.concatenate(old_log_probs)
            
            for g in range(gradient_epochs):
                early_stop = False
                idx = np.random.permutation(len(all_states))
                for i in range(0, len(all_states), batch_size):
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

                    adv = all_advantages[indices]
                    ret = all_returns[indices]

                    new_probs, new_values = self.nn.forward_prop(s, d, l, du, dr, dd, dl, dx, dy)

                    action_indices = np.argmax(a, axis=1)

                    new_log_probs = np.log(
                        new_probs[np.arange(len(action_indices)), action_indices] + 1e-8
                    )

                    ratio = np.exp(new_log_probs - old_log_probs[indices])

                    clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
                    ppo_weight = np.minimum(ratio * adv, clipped_ratio * adv)
                    
                    # ppo_weight = (ppo_weight - ppo_weight.mean()) / (ppo_weight.std() + 1e-8)
                    
                    # entropy_bonus = 0.01 * -np.sum(new_probs * np.log(new_probs + 1e-8), axis=1)
                    # ppo_weight += entropy_bonus

                    ret_mean = ret.mean()
                    ret_std  = ret.std() + 1e-8
                    critic_loss_signal = (ret - ret_mean) / ret_std
                    
                    if np.mean(np.abs(ratio - 1.0)) > 0.5:
                        early_stop = True
                        break

                    self.nn.backward_prop(a, ppo_weight, critic_loss_signal, learning_rate)
                    
                    # raw_avg = snake_length_sum / episodes
                    # print(f"Epoch {e}: avg_length={raw_avg:.2f}, grad_epochs_run={g+1}")
                if early_stop:
                    break
        epoch_avg /= epochs
        return epoch_avg