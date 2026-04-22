import numpy as np
from scipy import signal

class Convolution():
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (kernel_size * kernel_size * input_depth))
        self.biases = np.zeros(self.output_shape)
    
    def forward_prop(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        self.pre_activation_output = self.output
        self.output = self.ReLu(self.output)
        return self.output
    
    def backward_prop(self, output, learning_rate=0.001):
        output_gradient = output * self.Derivative_ReLu(self.pre_activation_output)
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i , j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        bias_gradient = np.sum(output_gradient, axis=(1, 2))
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * bias_gradient[:, None, None]
        return input_gradient
    
    def ReLu(self, z):
        return np.maximum(0, z)
    
    def Derivative_ReLu(self, z):
        return (z > 0).astype(np.float32)
    
    def save(self, filename):
        np.savez(
            filename,
            kernels=self.kernels,
            biases=self.biases,
            input_depth=self.input_depth,
            depth=self.depth,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            kernels_shape=self.kernels_shape
        )
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.kernels = data["kernels"]
        self.biases = data["biases"]
        self.input_depth = data["input_depth"]
        self.depth = data["depth"]
        self.input_shape = data["input_shape"]
        self.output_shape = data["output_shape"]
        self.kernels_shape = data["kernels_shape"]