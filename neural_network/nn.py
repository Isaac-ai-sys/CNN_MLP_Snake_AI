import numpy as np
from convolution import Convolution
from dense import Dense
from reshape import Reshape

class NN():
    def __init__(self):
        self.layers = []
    
    def forward_prop(self, input, direction, length):
        for layer in self.layers[:-1]:
            if isinstance(layer, Reshape):
                input = layer.forward_prop(input, direction length)
            else:
                input = layer.forward_prop(input)
        return self.layers[-1].forward_prop_softmax(input)
    
    def backward_prop(self, input, advantage):
        input = self.layers[-1].backward_prop_softmax(input, advantage)
        for layer in reversed(self.layers[:-1]):
            input = layer.backward_prop(input)
        return input
    
    def add_reshape_layer(self, input_shape, output_shape):
        self.layers.append(Reshape(input_shape, output_shape))
    
    def add_dense_layer(self, neurons, inputs):
        self.layers.append(Dense(neurons, inputs))
    
    def add_convolution_layer(self, input_shape, kernel_size, depth):
        self.layers.append(Convolution(input_shape, kernel_size, depth))
    
    def save(self):
        for i in range(len(self.layers)):
            self.layers[i].save(f"Models/layer{i}")
    
    def load(self):
        for i in range(len(self.layers)):
            self.layers[i].load(f"Models/layer{i}.npz")