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
                while env.running:
                    state, direction, length = env.get_state()
                    states.append(state)
                    directions.append(direction)
                    lengths.append(length)
                    
                    action = self.nn.choose_action(state, direction, length)
                    actions.append(action)
                    
                    reward = env.step(action)
                    rewards.append(reward)
                
                