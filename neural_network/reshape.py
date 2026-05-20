import numpy as np

class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape       # static conv output shape (no batch)
        self.output_shape = output_shape
        self.batch_input_shape = None        # set during forward, includes batch dim
    
    def forward_prop(self, input, direction, length, danger_up, danger_right, danger_down, danger_left, dx_food, dy_food):
        self.batch_input_shape = input.shape
        batch_size = input.shape[0]
        
        flat_input = input.reshape(batch_size, -1)
        flat_direction = direction.reshape(batch_size, -1)
        
        extras = np.stack([
            np.atleast_1d(length),
            np.atleast_1d(danger_up),
            np.atleast_1d(danger_right),
            np.atleast_1d(danger_down),
            np.atleast_1d(danger_left),
            np.atleast_1d(dx_food),
            np.atleast_1d(dy_food)
        ], axis=1)  # shape: (batch_size, 7)
        
        return np.concatenate([flat_input, flat_direction, extras], axis=1)
    
    def backward_prop(self, output_gradient, learning_rate=0.001):
        input_size = np.prod(self.batch_input_shape[1:])  # C*H*W
        grad_input = output_gradient[:, :input_size]
        return grad_input.reshape(self.batch_input_shape)  # (N, C, H, W)
    
    def save(self, filename):
        np.savez(filename,
                 input_shape=self.input_shape,
                 output_shape=self.output_shape,
                 batch_input_shape = self.batch_input_shape)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]
        self.batch_input_shape = data["batch_input_shape"]