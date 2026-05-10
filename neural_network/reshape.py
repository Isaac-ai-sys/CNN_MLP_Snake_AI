import numpy as np

class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward_prop(self, input, direction, length):
        self.input_shape = input.shape
        batch_size = self.input_shape[0]
        
        flat_input = input.reshape(batch_size, -1)
        flat_direction = direction.reshape(batch_size, -1)
        length_col = np.full((batch_size, 1), length)
        
        return np.concatenate(
            [flat_input, flat_direction, length_col],
            axis=1
        )
    
    def backward_prop(self, output_gradient, learning_rate=0.001):
        input_size = np.prod(self.input_shape[1:])
        
        grad_input = output_gradient[:, :input_size]
        
        return grad_input.reshape(self.input_shape)
    
    def save(self, filename):
        np.savez(filename,
                 input_shape=self.input_shape,
                 output_shape=self.output_shape)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]