import numpy as np
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net, board_size=20):
        self.nn = neural_net
        self.board_size = board_size
    
    def train(self, epochs=100, episodes=64, max_steps=1000, learning_rate=0.001):
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

            avg = (snake_length_sum / episodes) * size * size
            print(f"AVG Snake Length for epoch {e}: {avg:.4f}")

            all_states = np.concatenate([ep[0] for ep in episode_data])
            all_directions = np.concatenate([ep[1] for ep in episode_data])
            all_lengths = np.concatenate([ep[2] for ep in episode_data])
            all_actions = np.concatenate([ep[3] for ep in episode_data])
            all_advantages = np.concatenate([ep[4] for ep in episode_data])

            all_advantages = all_advantages - all_advantages.mean()  # baseline only, no scaling

            # Diagnostic first (safe, doesn't affect training)
            probs = self.nn.forward_prop(all_states[:5], all_directions[:5], all_lengths[:5])
            print(f"Sample probs: {probs[0].round(3)}  entropy: {-np.sum(probs * np.log(probs + 1e-8), axis=1).mean():.3f}")
            
            self.nn.forward_prop(all_states, all_directions, all_lengths)
            self.nn.backward_prop(all_actions, all_advantages)
        return