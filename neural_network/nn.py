import numpy as np
from neural_network.convolution import Convolution
from neural_network.dense import Dense
from neural_network.reshape import Reshape
from neural_network.max_pool import Max_Pool

class NN():
    def __init__(self):
        self.actor_layers = []
        self.critic_layers = []
        self.feature_layers = []
        from pathlib import Path

        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.save_dir = self.BASE_DIR / "Models"
        (self.save_dir / "actor").mkdir(exist_ok=True)
        (self.save_dir / "critic").mkdir(exist_ok=True)
        (self.save_dir / "feature").mkdir(exist_ok=True)
    
    def forward_prop(self, state, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food):
        # feature forward prop
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food)
            else:
                feature_input = layer.forward_prop(feature_input)
        features = feature_input
        
        # actor forward prop
        actor_input = features
        for layer in self.actor_layers[:-1]:
            actor_input = layer.forward_prop(actor_input)
        actor_probs = self.actor_layers[-1].forward_prop_softmax(actor_input)
        
        # critic forward prop
        critic_input = features
        for layer in self.critic_layers[:-1]:
            critic_input = layer.forward_prop(critic_input)
        critic_value = self.critic_layers[-1].forward_prop_value(critic_input)
        
        return actor_probs, critic_value
    
    def backward_prop(self, actions_one_hot, ppo_weight, target, learning_rate=0.001):
        # actor back_prop
        actor_input = actions_one_hot
        actor_input = self.actor_layers[-1].backward_prop_softmax(actor_input, ppo_weight, learning_rate)
        for layer in reversed(self.actor_layers[:-1]):
            actor_input = layer.backward_prop(actor_input, learning_rate)
        actor_dx = actor_input
        
        # value back_prop
        critic_input = target
        critic_input = self.critic_layers[-1].backward_prop_value(critic_input, learning_rate)
        for layer in reversed(self.critic_layers[:-1]):
            critic_input = layer.backward_prop(critic_input, learning_rate)
        critic_dx = critic_input
        
        # feature back_prop
        actor_dx = -actor_dx  # negate for feature layers
        actor_dx /= np.std(actor_dx) + 1e-8
        critic_dx /= np.std(critic_dx) + 1e-8
        feature_input = actor_dx + critic_dx
        feature_input /= np.std(feature_input) + 1e-8
        # print(f"feature input std (entering feature layers): {np.std(feature_input):.6f}")  # should be ~1.0
        for layer in reversed(self.feature_layers):
            feature_input = layer.backward_prop(feature_input, learning_rate)
        
        # print(f"actor_dx std: {np.std(actor_dx):.6f}")
        # print(f"critic_dx std: {np.std(critic_dx):.6f}")
        # print(f"feature input std: {np.std(feature_input):.6f}")
        return feature_input
    
    def create_reshape_layer(self, input_shape, output_shape):
        return Reshape(input_shape, output_shape)
    
    def create_dense_layer(self, neurons, inputs):
        return Dense(neurons, inputs)
    
    def create_convolution_layer(self, input_shape, kernel_size, depth):
        return Convolution(input_shape, kernel_size, depth)
    
    def create_max_pool_layer(self, size):
        return Max_Pool(size)
    
    def save(self):
        for i in range(len(self.actor_layers)):
            self.actor_layers[i].save(f"{self.save_dir}/actor/layer{i}")
        for i in range(len(self.critic_layers)):
            self.critic_layers[i].save(f"{self.save_dir}/critic/layer{i}")
        for i in range(len(self.feature_layers)):
            self.feature_layers[i].save(f"{self.save_dir}/feature/layer{i}")
    
    def load(self):
        for i in range(len(self.actor_layers)):
            self.actor_layers[i].load(f"{self.save_dir}/actor/layer{i}.npz")
        for i in range(len(self.critic_layers)):
            self.critic_layers[i].load(f"{self.save_dir}/critic/layer{i}.npz")
        for i in range(len(self.feature_layers)):
            self.feature_layers[i].load(f"{self.save_dir}/feature/layer{i}.npz")
    
    def choose_action(self, input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food):
        # ensure single sample
        input = np.array(input)
        direction = np.array(direction)

        if input.ndim == 1:
            input = input[None, :]
            direction = direction[None, :]

        probs, values = self.forward_prop(input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food)
        
        # extract first (and only) sample
        probs = probs[0]
        value = values[0]

        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        result = np.zeros(len(probs))
        result[action] = 1

        return result, value, log_prob