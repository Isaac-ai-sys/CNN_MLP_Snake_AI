import numpy as np
from nn import NN
from ..game.snake_env import Snake_Env

class Train():
    def __init__(self, neural_net):
        self.nn = neural_net
    
    def train(self, epochs=10, episodes=100, max_steps=100, learning_rate=0.001):
        for e in epochs:
            snake_length_sum = 0
            for ep in episodes:
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
                for i in range(len(rewards)):
                    rewards[i] *= gamma
                    gamma *= 0.99
                
                