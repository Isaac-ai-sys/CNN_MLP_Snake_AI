import numpy as np
from game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net):
        self.nn = neural_net
    
    def train(self, epochs=10, episodes=64, max_steps=100, learning_rate=0.001):
        for e in range(epochs):
            snake_length_sum = 0
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_directions = []
            batch_lengths = []
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
                
                # discount rewards
                gamma = .99
                discounted_returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    discounted_returns.insert(0, G)
                
                snake_length_sum += lengths[-1]
                
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_directions.extend(directions)
                batch_lengths.extend(lengths)

            avg = (snake_length_sum * env.size * env.size) / episodes
            print(f"AVG Snake Length for epoch {e}:  {avg}")
            
            batch_states = np.array(batch_states)
            batch_actions = np.array(batch_actions)
            batch_rewards = np.array(batch_rewards)
            batch_directions = np.array(batch_directions)
            batch_lengths = np.array(batch_lengths)
            
            batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-8)
            
            self.nn.forward_prop(batch_states, batch_directions, batch_lengths)
            self.nn.backward_prop(batch_actions, batch_rewards)
        return