import numpy as np

class Dense():
    def __init__(self, neurons, inputs):
        self.weights = np.random.randn(neurons, inputs) * 0.01
        self.biases = np.zeros(neurons)
    
    def ReLu(self, z):
        return np.maximum(0, z)
    
    def derivative_ReLu(self, z):
        return (z > 0).astype(np.float32)
    
    def softmax(self, z):
        z = z - np.max(z)
        exp = np.exp(z)
        return exp / (np.sum(exp) + 1e-12)
    
    def forward_prop(self, input):
        x = np.asarray(input)
        
        self.input = x
        self.pre_activated = self.weights.dot(self.input) + self.biases
        self.output = self.ReLu(self.pre_activated)
        return self.output
    
    def forward_prop_softmax(self, input):
        x = np.asarray(input)
    
        self.input = x
        self.pre_activated_output = self.weights.dot(self.input) + self.biases
        self.output = self.softmax(self.pre_activated_output)
        return self.output
    
    def backward_prop_softmax(self, y_true, advantage, learning_rate=0.001):
        dz = self.output - y_true
        dz *= advantage
        
        dw = np.outer(dz, self.input)
        db = dz
        dx = self.weights.T.dot(dz)
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return dx
    
    def backward_prop(self, da, learning_rate=0.001):
        dz = da * self.derivative_ReLu(self.pre_activated)
        dw = np.outer(dz, self.input)
        db = dz
        dx = self.weights.T.dot(dz)
        
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
    
    def get_params(self):
        return [self.weights, self.biases]

    def set_params(self, params):
        
        self.weights = params[0]
        self.biases = params[1]