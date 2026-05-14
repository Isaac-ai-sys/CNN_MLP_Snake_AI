import numpy as np
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net, board_size=20):
        self.nn = neural_net
        self.board_size = board_size
    
    def train(self, epochs=100, episodes=100, max_steps=1000, learning_rate=0.001):
        epoch_avg = 0
        entropy_avg = 0
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            for ep in range(episodes):
                states, actions, rewards, directions, lengths = [], [], [], [], []
                env = Snake_Env(self.board_size)
                size = env.size
                step = 0
                while env.running and step < max_steps:
                    state, direction, length = env.get_state()
                    states.append(state)
                    directions.append(direction)
                    lengths.append(length)
                    action = self.nn.choose_action(state, direction, length)
                    actions.append(action)
                    rewards.append(env.step(action))
                    step += 1

                gamma = 0.99
                G = 0
                discounted_returns = []
                for r in reversed(rewards):
                    G = r + gamma * G
                    discounted_returns.insert(0, G)
                
                snake_length_sum += lengths[-1]
                episode_data.append((states, directions, lengths, actions, discounted_returns))
                #print(discounted_returns)

            avg = (snake_length_sum * size * size) / episodes
            epoch_avg += avg
            # print(f"AVG Snake Length for epoch {e}: {avg:.4f}")
            
            all_states = np.concatenate([ep[0] for ep in episode_data])
            all_directions = np.concatenate([ep[1] for ep in episode_data])
            all_lengths = np.array([l for ep in episode_data for l in ep[2]])
            all_actions = np.concatenate([ep[3] for ep in episode_data])
            all_advantages = []

            for i, ep in enumerate(episode_data):
                returns = ep[4]  # discounted_returns
                adv = returns - np.mean(returns)
                all_advantages.append(adv)

            # concatenate across episodes
            all_advantages = np.concatenate(all_advantages)
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
            all_advantages = np.clip(all_advantages, -5, 5)
            
            # for advantage in all_advantages:
            #     print(advantage)
            
            # diagnostic first (safe, doesn't affect training)
            probs = self.nn.forward_prop(all_states, all_directions, all_lengths)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
            entropy_avg += entropy
            # print(f"entropy: {entropy:.3f}")
            
            # print("adv mean:", all_advantages.mean())
            # print("adv std:", all_advantages.std())
            
            batch_size = 64
            for i in range(0, len(all_states), batch_size):
                s = all_states[i:i+batch_size]
                d = all_directions[i:i+batch_size]
                l = all_lengths[i:i+batch_size]
                a = all_actions[i:i+batch_size]
                adv = all_advantages[i:i+batch_size]

                self.nn.forward_prop(s, d, l)
                self.nn.backward_prop(a, adv, learning_rate)
        epoch_avg /= epochs
        entropy_avg /= epochs
        # print(f"epoch_avg: {epoch_avg:.3f}")
        # print(f"entropy_avg: {entropy_avg:.3f}")
        return epoch_avg, entropy_avg