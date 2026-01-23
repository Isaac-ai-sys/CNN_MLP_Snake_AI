import numpy as np

class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward_prop(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward_prop(self, output_gradient, learning_rate=0.001):
        return np.reshape(output_gradient, self.input_shape)
    
    def save(self, filename):
        np.savez(filename,
                 input_shape=self.input_shape,
                 output_shape=self.output_shape)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]