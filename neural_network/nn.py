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
        self.value_layers = []
        from pathlib import Path

        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.save_dir = self.BASE_DIR / "Models"
        (self.save_dir / "actor").mkdir(exist_ok=True)
        (self.save_dir / "critic").mkdir(exist_ok=True)
        (self.save_dir / "feature").mkdir(exist_ok=True)
        # optimizer setting: 'sgd' or 'adam'
        self.optimizer = 'adam'
    
    def forward_prop(self, state, direction, length, dx_food, dy_food, running):
        # feature forward prop - run once
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, dx_food, dy_food, running)
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
    
    def forward_prop_value(self, state, direction, length, dx_food, dy_food, running):
        # reuse existing feature forward prop
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, dx_food, dy_food, running)
            else:
                feature_input = layer.forward_prop(feature_input)
        
        value_input = feature_input
        for layer in self.value_layers[:-1]:
            value_input = layer.forward_prop(value_input)
        return self.value_layers[-1].forward_prop_value(value_input)
    
    def forward_prop_search(self, state, direction, length, dx_food, dy_food, running):
        # feature forward prop - run once
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, dx_food, dy_food, running)
            else:
                feature_input = layer.forward_prop(feature_input)
        features = feature_input
        
        # actor forward prop
        actor_input = features
        for layer in self.actor_layers[:-1]:
            actor_input = layer.forward_prop(actor_input)
        actor_probs = self.actor_layers[-1].forward_prop_softmax(actor_input)
        
        # value forward prop
        value_input = feature_input
        for layer in self.value_layers[:-1]:
            value_input = layer.forward_prop(value_input)
        value_output = self.value_layers[-1].forward_prop_value(value_input)
        
        return actor_probs, value_output
    
    def forward_prop_full(self, state, direction, length, dx_food, dy_food, running):
        # feature forward prop - run once
        feature_input = state
        for layer in self.feature_layers:
            if isinstance(layer, Reshape):
                feature_input = layer.forward_prop(feature_input, direction, length, dx_food, dy_food, running)
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
        
        # value forward prop
        value_input = feature_input
        for layer in self.value_layers[:-1]:
            value_input = layer.forward_prop(value_input)
        value_output = self.value_layers[-1].forward_prop_value(value_input)
        
        return actor_probs, critic_value, value_output
    
    def backward_prop(self, actions_one_hot, advantages, returns, actor_learning_rate=0.0001, critic_learning_rate=0.0001, entropy_beta=0.02, value_loss_coef=0.5, optimizer=None):
        # actor back_prop
        actor_input = actions_one_hot
        opt = optimizer if optimizer is not None else self.optimizer
        actor_input = self.actor_layers[-1].backward_prop_softmax(actor_input, advantages, actor_learning_rate, entropy_beta, optimizer=opt)
        for layer in reversed(self.actor_layers[:-1]):
            actor_input = layer.backward_prop(actor_input, actor_learning_rate, optimizer=opt)
        actor_dx = actor_input
        
        # value back_prop
        critic_input = returns
        critic_input = self.critic_layers[-1].backward_prop_value(critic_input, critic_learning_rate, value_loss_coef, optimizer=opt)
        for layer in reversed(self.critic_layers[:-1]):
            critic_input = layer.backward_prop(critic_input, critic_learning_rate, optimizer=opt)
        # critic_dx = critic_input
        
        # feature back_prop
        # combine actor and critic gradients for shared feature layers
        # avoid amplifying tiny stddevs (which can destabilize training)
        actor_std = np.std(actor_dx)
        # critic_std = np.std(critic_dx)
        min_std = 1e-2
        actor_dx /= max(actor_std, min_std)
        # critic_dx /= max(critic_std, min_std)
        # nn.py — current
        feature_input = actor_dx # + 0.5 * critic_dx could add this back in
        # feature_input = np.clip(feature_input, -1, 1)
        feature_input /= max(np.std(feature_input), min_std)
        # print(f"feature input std (entering feature layers): {np.std(feature_input):.6f}")  # should be ~1.0
        # print(f"gradient entering feature layers std: {np.std(feature_input):.6f}")
        for layer in reversed(self.feature_layers):
            # feature layers may not have optimizer-specific updates, but accept optimizer param
            try:
                feature_input = layer.backward_prop(feature_input, actor_learning_rate, optimizer=opt)
            except TypeError:
                feature_input = layer.backward_prop(feature_input, actor_learning_rate)
            # print(f"  after {type(layer).__name__}: std={np.std(feature_input):.6f}")
        
        # print(f"actor_dx std: {np.std(actor_dx):.6f}")
        # print(f"critic_dx std: {np.std(critic_dx):.6f}")
        # print(f"feature input std: {np.std(feature_input):.6f}")
        return feature_input
    
    def backward_prop_value_head(self, targets, learning_rate=0.0001):
        dx = self.value_layers[-1].backward_prop_value(targets, learning_rate)
        for layer in reversed(self.value_layers[:-1]):
            dx = layer.backward_prop(dx, learning_rate)
        return dx
    
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
        for i in range(len(self.value_layers)):
            self.value_layers[i].save(f"{self.save_dir}/value/layer{i}")
    
    def _load_layer(self, layer, filename):
        from pathlib import Path
        path = Path(filename)
        if not path.exists():
            return

        old_params = {}
        for attr in ("weights", "biases", "kernels"):
            if hasattr(layer, attr):
                old_params[attr] = getattr(layer, attr)

        try:
            layer.load(str(path))

            for attr, old_value in old_params.items():
                new_value = getattr(layer, attr, None)
                if new_value is not None and new_value.shape != old_value.shape:
                    raise ValueError(
                        f"Loaded '{path}' has {attr} shape {new_value.shape}, "
                        f"expected {old_value.shape}"
                    )
        except Exception as exc:
            print(
                f"Warning: could not load '{path}': {exc}. "
                "Using randomly initialized layer instead."
            )
            for attr, old_value in old_params.items():
                setattr(layer, attr, old_value)

    def load(self):
        for i in range(len(self.actor_layers)):
            self._load_layer(self.actor_layers[i], f"{self.save_dir}/actor/layer{i}.npz")
        for i in range(len(self.critic_layers)):
            self._load_layer(self.critic_layers[i], f"{self.save_dir}/critic/layer{i}.npz")
        for i in range(len(self.feature_layers)):
            self._load_layer(self.feature_layers[i], f"{self.save_dir}/feature/layer{i}.npz")
        for i in range(len(self.value_layers)):
            self._load_layer(self.value_layers[i], f"{self.save_dir}/value/layer{i}.npz")
    
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