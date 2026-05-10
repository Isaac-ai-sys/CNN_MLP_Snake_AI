import numpy as np

class Dense():
    def __init__(self, neurons, inputs):
        self.weights = np.random.randn(neurons, inputs) * np.sqrt(2 / inputs)
        self.biases = np.zeros(neurons)
    
    def ReLu(self, pre_activated):
        return np.maximum(0, pre_activated)
    
    def derivative_ReLu(self, pre_activated):
        return (pre_activated > 0).astype(np.float32)
    
    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def forward_prop(self, input):
        if input.ndim == 1:
            input = input[None, :]
        self.input = input
        self.pre_activated = self.input @ self.weights.T + self.biases
        self.output = self.ReLu(self.pre_activated)
        return self.output
    
    def forward_prop_softmax(self, input):
        if input.ndim == 1:
            input = input[None, :]
        self.input = input
        self.pre_activated_output = self.input @ self.weights.T + self.biases
        self.output = self.softmax(self.pre_activated_output)
        return self.output
    
    def backward_prop_softmax(self, y_true, advantage, learning_rate=0.001):
        batch_size = self.input.shape[0]
        dz = self.output - y_true
        dz *= advantage[:, None]
        
        dw = (dz.T @ self.input) / batch_size
        db = np.mean(dz, axis=0)
        dx = dz @ self.weights
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return dx
    
    def backward_prop(self, da, learning_rate=0.001):
        batch_size = self.input.shape[0]
        dz = da * self.derivative_ReLu(self.pre_activated)
        dw = (dz.T @ self.input) / batch_size
        db = np.mean(dz, axis=0)
        dx = dz @ self.weights
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return dx
    
    def save(self, filename):
        np.savez(filename,
                 weights=self.weights,
                 biases=self.biases)
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.weights = data["weights"]
        self.biases = data["biases"]