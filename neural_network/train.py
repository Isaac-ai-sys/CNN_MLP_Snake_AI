import numpy as np
from neural_network.nn import NN
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net):
        self.nn = neural_net
    
    def train(self, epochs=10, episodes=100, max_steps=100, learning_rate=0.001):
        for e in range(epochs):
            snake_length_sum = 0
            for ep in range(episodes):
                states = []
                actions = []
                rewards = []
                directions = []
                lengths = []
                
                env = Snake_Env()
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
                
                # normalize rewards
                gamma = .99
                discounted_returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    discounted_returns.insert(0, G)
                #normalize rewards
                rewards = np.array(discounted_returns)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                
                for i in indices:
                    self.nn.forward_prop(states[i], directions[i], lengths[i])
                    self.nn.backward_prop(actions[i], rewards[i])
                
                snake_length_sum += lengths[-1]
            avg = (snake_length_sum * env.size * env.size) / episodes
            print(f"AVG Snake Length for epoch {e}:  {avg}")
        return