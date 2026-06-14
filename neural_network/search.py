import numpy as np
import copy
from neural_network.nn import NN
from game.snake_env import VectorizedSnakeEnv

class search():
    def __init__(self, neural_net, depth=1):
        self.nn = neural_net
        self.depth = depth
    
    #DFS search algorithm that checks top two moves from nn in each position and evaluates position using critic
    def find_best_action(self, state, direction, length, dx_food, dy_food, running, depth, env):
        probs, value = self.nn.forward_prop(state, direction, length, dx_food, dy_food, running)

        if depth >= self.depth or running[0] != 1:
            if running[0] != 1:
                return None, -1e6
            return None, float(value[0, 0])

        action_1_one_hot, action_2_one_hot = self.find_top_two_actions(probs[0])
        action_1_idx = int(action_1_one_hot.argmax())
        action_2_idx = int(action_2_one_hot.argmax())

        env_1 = copy.deepcopy(env)
        env_2 = copy.deepcopy(env)

        env_1.step(np.array([action_1_idx]))
        state_1, direction_1, length_1, dx_food_1, dy_food_1, running_1 = env_1.get_state()

        env_2.step(np.array([action_2_idx]))
        state_2, direction_2, length_2, dx_food_2, dy_food_2, running_2 = env_2.get_state()

        _, left_value = self.find_best_action(state_1, direction_1, length_1, dx_food_1, dy_food_1, running_1, depth + 1, env_1)
        _, right_value = self.find_best_action(state_2, direction_2, length_2, dx_food_2, dy_food_2, running_2, depth + 1, env_2)

        if left_value >= right_value:
            return action_1_one_hot, left_value
        else:
            return action_2_one_hot, right_value
    
    def find_top_two_actions(self, probs):
        top2 = np.argsort(probs)[-2:]
        action_1_one_hot = np.zeros(4)
        action_2_one_hot = np.zeros(4)
        idx1 = int(top2[1])
        idx2 = int(top2[0])
        action_1_one_hot[idx1] = 1
        action_2_one_hot[idx2] = 1
        return action_1_one_hot, action_2_one_hot