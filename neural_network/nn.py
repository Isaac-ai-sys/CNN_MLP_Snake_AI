try:
    import cupy as np
except:
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
    
    def forward_prop(self, state, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food, running):
        # feature forward prop
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food, running)
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
    
    def backward_prop(self, actions_one_hot, ppo_weight, target, actor_learning_rate=0.0001, critic_learning_rate=0.0001, entropy_beta=0.02, value_loss_coef=0.5):
        # actor back_prop
        actor_input = actions_one_hot
        actor_input = self.actor_layers[-1].backward_prop_softmax(actor_input, ppo_weight, actor_learning_rate, entropy_beta)
        for layer in reversed(self.actor_layers[:-1]):
            actor_input = layer.backward_prop(actor_input, actor_learning_rate)
        actor_dx = actor_input
        
        # value back_prop
        critic_input = target
        critic_input = self.critic_layers[-1].backward_prop_value(critic_input, critic_learning_rate, value_loss_coef)
        for layer in reversed(self.critic_layers[:-1]):
            critic_input = layer.backward_prop(critic_input, critic_learning_rate)
        critic_dx = critic_input
        
        # feature back_prop
        # combine actor and critic gradients for shared feature layers
        # avoid amplifying tiny stddevs (which can destabilize training)
        actor_std = np.std(actor_dx)
        critic_std = np.std(critic_dx)
        min_std = 1e-2
        actor_dx /= max(actor_std, min_std)
        critic_dx /= max(critic_std, min_std)
        # nn.py — current
        feature_input = actor_dx + 0.5 * critic_dx
        # feature_input = np.clip(feature_input, -1, 1)
        feature_input /= max(np.std(feature_input), min_std)
        # print(f"feature input std (entering feature layers): {np.std(feature_input):.6f}")  # should be ~1.0
        # print(f"gradient entering feature layers std: {np.std(feature_input):.6f}")
        for layer in reversed(self.feature_layers):
            feature_input = layer.backward_prop(feature_input, actor_learning_rate)
            # print(f"  after {type(layer).__name__}: std={np.std(feature_input):.6f}")
        
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
    
    def choose_actions_batch(self, input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food, running, epsilon=0.0):
        input = np.asarray(input)
        direction = np.asarray(direction)

        if input.ndim == 3:
            input = input[None, :]
            direction = direction[None, :]

        batch_size = input.shape[0]
        probs, values = self.forward_prop(input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food, running)

        cumprobs = probs.cumsum(axis=1)
        rand = np.random.rand(batch_size, 1).astype(probs.dtype)
        action_indices = np.argmax(cumprobs >= rand, axis=1)

        if epsilon > 0.0:
            random_actions = np.random.randint(0, probs.shape[1], size=batch_size)
            explore = np.random.rand(batch_size) < epsilon
            action_indices = np.where(explore, random_actions, action_indices)

        action_one_hot = np.zeros_like(probs)
        batch_idx = np.arange(batch_size)
        action_one_hot[batch_idx, action_indices] = 1

        log_probs = np.log(probs[batch_idx, action_indices] + 1e-8)
        return action_one_hot, action_indices, values, log_probs

    def choose_action(self, input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food, running, epsilon=0.0):
        # ensure single sample
        input = np.array(input)
        direction = np.array(direction)

        if input.ndim == 1:
            input = input[None, :]
            direction = direction[None, :]

        action_one_hot, action_indices, values, log_probs = self.choose_actions_batch(
            input,
            direction,
            length,
            danger_up,
            danger_right,
            danger_down,
            danger_left,
            dx_food,
            dy_food,
            running,
            epsilon
        )
        action = action_one_hot[0]
        value = values[0]
        log_prob = log_probs[0]

        # GPU -> CPU
        to_cpu = lambda x: x.get() if hasattr(x, "get") else x

        return (
            to_cpu(action),
            to_cpu(value),
            to_cpu(log_prob)
        )