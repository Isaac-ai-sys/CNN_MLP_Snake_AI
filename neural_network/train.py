import numpy as np
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net):
        self.nn = neural_net
    
    def train(self, epochs=10, episodes=64, max_steps=2000, learning_rate=0.001):
        for e in range(epochs):
            episode_data = []
            snake_length_sum = 0
            for ep in range(episodes):
                
                states = []
                actions = []
                rewards = []
                directions = []
                lengths = []
                
                env = Snake_Env()
                size = env.size
                step = 0
                while env.running and step < max_steps:
                    state, direction, length = env.get_state()
                    states.append(state)
                    directions.append(direction)
                    lengths.append(length)
                    
                    action = self.nn.choose_action(state, direction, length)
                    actions.append(action)
                    
                    reward = env.step(action)
                    rewards.append(reward)
                    step += 1
                
                # discount rewards
                gamma = .99
                discounted_returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    discounted_returns.insert(0, G)
                
                snake_length_sum += lengths[-1]
                
                discounted_returns = np.array(discounted_returns)
                discounted_returns = discounted_returns - np.mean(discounted_returns)
                discounted_returns = discounted_returns / (np.std(discounted_returns) + 1e-8)
                
                episode_data.append((states, directions, lengths, actions, discounted_returns))

            avg = size * size * snake_length_sum / episodes
            print(f"AVG Snake Length for epoch {e}:  {avg}")
            
            for states, directions, lengths, actions, advantages in episode_data:
                states = np.array(states)
                directions = np.array(directions)
                lengths = np.array(lengths)
                actions = np.array(actions)
                advantages = np.array(advantages)
                
                self.nn.forward_prop(states, directions, lengths)
                self.nn.backward_prop(actions, advantages)
        return