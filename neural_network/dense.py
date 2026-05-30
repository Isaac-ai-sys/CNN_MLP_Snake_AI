try:
    import cupy as np
except:
    import numpy as np

class Dense():
    def __init__(self, neurons, inputs):
        self.weights = np.random.randn(neurons, inputs) * np.sqrt(2 / inputs)
        self.biases = np.zeros(neurons)
        self.last_dw_norm = 0.0
        self.last_db_norm = 0.0
    
    def ReLu(self, pre_activated):
        return np.where(pre_activated > 0, pre_activated, 0.01 * pre_activated)

    def derivative_ReLu(self, pre_activated):
        return np.where(pre_activated > 0, 1.0, 0.01).astype(np.float32)
    
    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def forward_prop(self, input):
        if input.ndim == 1:
            input = input[None, :]
        input = np.asarray(input)
        self.input = input
        self.pre_activated = self.input @ self.weights.T + self.biases
        self.output = self.ReLu(self.pre_activated)
        return self.output
    
    def forward_prop_softmax(self, input):
        if input.ndim == 1:
            input = input[None, :]
        input = np.asarray(input)
        self.input = input
        self.pre_activated_output = self.input @ self.weights.T + self.biases
        self.output = self.softmax(self.pre_activated_output)
        return self.output
    
    def forward_prop_value(self, input):
        if input.ndim == 1:
            input = input[None, :]
        input = np.asarray(input)
        self.input = input
        self.pre_activated_output = self.input @ self.weights.T + self.biases
        self.output = self.pre_activated_output
        return self.output

    def backward_prop_value(self, target, learning_rate=0.001, value_loss_coef=1.0, max_grad_norm=0.5):
        target = np.asarray(target)
        batch_size = self.input.shape[0]
        target = target.reshape(-1, 1)
        self.output = self.output.reshape(-1, 1)
        
        dz = self.output - target
        dz *= value_loss_coef
        # dz /= (np.std(target) + 1e-8)
        
        dw = (dz.T @ self.input) / batch_size
        db = np.mean(dz, axis=0)
        dx = dz @ self.weights
        
        # record parameter gradient norms for diagnostics
        try:
            self.last_dw_norm = float(np.linalg.norm(dw))
            self.last_db_norm = float(np.linalg.norm(db))
        except Exception:
            self.last_dw_norm = 0.0
            self.last_db_norm = 0.0

        # compute global norm
        grad_norm = np.sqrt(np.sum(dw**2) + np.sum(db**2))

        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-8)

            dw *= scale
            db *= scale
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        return dx
    
    def backward_prop_softmax(self, actions_one_hot, ppo_weight, learning_rate=0.0001, entropy_beta=0.02, max_grad_norm=0.5):
        batch_size = self.input.shape[0]
        actions_one_hot = np.asarray(actions_one_hot)
        ppo_weight = np.asarray(ppo_weight)
        
        dz = self.output * ppo_weight[:, None]
        batch_idx = np.arange(batch_size)
        action_idx = np.argmax(actions_one_hot, axis=1)

        dz[batch_idx, action_idx] -= ppo_weight

        if entropy_beta != 0.0:
            # gradient of entropy H = -sum p log p w.r.t. logits z is: dH/dz_j = p_j * (H - log p_j)
            # for loss = actor_loss - entropy_coef * H, add: -entropy_coef * dH/dz = entropy_coef * p * (log p - H)
            entropy = -np.sum(self.output * np.log(self.output + 1e-8), axis=1)
            dz += entropy_beta * self.output * (np.log(self.output + 1e-8) + entropy[:, None])

        dw = (dz.T @ self.input) / batch_size
        db = np.mean(dz, axis=0)
        dx = dz @ self.weights

        # record parameter gradient norms for diagnostics
        try:
            self.last_dw_norm = float(np.linalg.norm(dw))
            self.last_db_norm = float(np.linalg.norm(db))
        except Exception:
            self.last_dw_norm = 0.0
            self.last_db_norm = 0.0

        # compute global norm
        grad_norm = np.sqrt(np.sum(dw**2) + np.sum(db**2))

        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-8)

            dw *= scale
            db *= scale
        
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        
        return dx
    def backward_prop(self, da, learning_rate=0.0001, max_grad_norm=0.5):
        batch_size = self.input.shape[0]
        da = np.asarray(da)
        
        dz = da * self.derivative_ReLu(self.pre_activated)
        
        dw = (dz.T @ self.input) / batch_size
        db = np.mean(dz, axis=0)
        dx = dz @ self.weights

        # record parameter gradient norms for diagnostics
        try:
            self.last_dw_norm = float(np.linalg.norm(dw))
            self.last_db_norm = float(np.linalg.norm(db))
        except Exception:
            self.last_dw_norm = 0.0
            self.last_db_norm = 0.0
        
        # compute global norm
        grad_norm = np.sqrt(np.sum(dw**2) + np.sum(db**2))

        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-8)

            dw *= scale
            db *= scale
        
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