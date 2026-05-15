import numpy as np
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net, board_size=20):
        self.nn = neural_net
        self.board_size = board_size
    
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        rewards = np.array(rewards)
        values = np.array(values).squeeze()

        values_next = np.append(values[1:], 0.0)

        deltas = rewards + gamma * values_next - values

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * lam * gae
            advantages[t] = gae

        return advantages
    
    def train(self, epochs=100, episodes=100, max_steps=1000, learning_rate=0.001):
        epoch_avg = 0
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            for ep in range(episodes):
                states, actions, values, rewards, directions, lengths, log_probs = [], [], [], [], [], [], []
                env = Snake_Env(self.board_size)
                size = env.size
                step = 0
                while env.running and step < max_steps:
                    state, direction, length = env.get_state()
                    states.append(state)
                    directions.append(direction)
                    lengths.append(length)
                    action, value, log_prob = self.nn.choose_action(state, direction, length)
                    actions.append(action)
                    values.append(value)
                    log_probs.append(log_prob)
                    rewards.append(env.step(action))
                    step += 1
                
                snake_length_sum += lengths[-1]
                episode_data.append((states, directions, lengths, actions, values, rewards, log_probs))
                #print(discounted_returns)

            avg = (snake_length_sum * size * size) / episodes
            epoch_avg += avg
            # print(f"AVG Snake Length for epoch {e}: {avg:.4f}")
            
            all_states = np.concatenate([ep[0] for ep in episode_data])
            all_directions = np.concatenate([ep[1] for ep in episode_data])
            all_lengths = np.array([l for ep in episode_data for l in ep[2]])
            all_actions = np.concatenate([ep[3] for ep in episode_data])
            all_values = np.concatenate([ep[4] for ep in episode_data])
            all_log_probs = np.concatenate([ep[6] for ep in episode_data])
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
            for i in range(0, len(all_states), batch_size):
                s = all_states[i:i+batch_size]
                d = all_directions[i:i+batch_size]
                l = all_lengths[i:i+batch_size]
                a = all_actions[i:i+batch_size]

                adv = all_advantages[i:i+batch_size]
                ret = all_returns[i:i+batch_size]

                new_probs, new_values = self.nn.forward_prop(s, d, l)

                action_indices = np.argmax(a, axis=1)

                new_log_probs = np.log(
                    new_probs[np.arange(len(action_indices)), action_indices] + 1e-8
                )

                ratio = np.exp(new_log_probs - all_log_probs[i:i+batch_size])

                clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
                ppo_weight = clipped_ratio * adv
                
                entropy_bonus = 0.01 * -np.sum(new_probs * np.log(new_probs + 1e-8), axis=1)
                ppo_weight += entropy_bonus
                
                ppo_weight = (ppo_weight - ppo_weight.mean()) / (ppo_weight.std() + 1e-8)

                critic_loss_signal = (ret - new_values.squeeze()) ** 2

                self.nn.backward_prop(a, ppo_weight, critic_loss_signal, learning_rate)
        epoch_avg /= epochs
        return epoch_avg