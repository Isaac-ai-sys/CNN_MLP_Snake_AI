import numpy as np
# from scipy import signal

class Convolution():
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (kernel_size * kernel_size * input_depth))
        self.biases = np.zeros((depth, 1, 1))
    
    # def forward_prop(self, input):
    #     # ensure shape is correct
    #     if input.ndim == 3:
    #         input = input[None, :]
    
    #     self.input = input
    #     batch_size = self.input.shape[0]
    #     self.output = np.zeros((batch_size, *self.output_shape))
    #     for b in range(batch_size):
    #         for i in range(self.depth):
    #             self.output[b, i] += self.biases[i, 0, 0]
    #             for j in range(self.input_depth):
    #                 self.output[b, i] += signal.correlate2d(self.input[b, j], self.kernels[i, j], "valid")
    #     self.pre_activation_output = self.output
    #     self.output = self.ReLu(self.output)
    #     return self.output
    
    def forward_prop(self, input):
        # ensure shape is correct
        if input.ndim == 3:
            input = input[None, :]
        
        self.input = input
        batch_size = self.input.shape[0]
        _, out_h, out_w = self.output_shape
        k = self.kernels_shape[2] #kernel size
        
        # step 1: extract every local path the kernel will land on
        # for each output position[i, j] grab the kxk region from each input channel
        # result shape: (batch_size, out_h, out_w, input_depth, k, k)
        patches = np.array([[input[:, :, i:i+k, j:j+k] for j in range(out_w)] for i in range(out_h)])
        
        #patches shape after squeeze: (out_h, out_w, batch_size, input_depth, k, k)
        patches = patches.transpose(2, 0, 1, 3, 4, 5) # (batch, out_h, out_w, input_depth, k, k)
        
        # step 2: flatten the (input_depth, k, k) patch into a single vector per position
        # each patch becomes a 1D vector of length input_depth * k * k
        # shape: (batch, out_h, out_w, input_depth * k * k)
        self.patches_flat = patches.reshape(batch_size, out_h, out_w, -1)
        
        # step 3: flatten each kernel the same way
        # kernels shape: (depth, input_depth, k, k) -> (depth, input_depth * k * k)
        self.kernels_flat = self.kernels.reshape(self.depth, -1)
        
        # step 4: matrix multiply
        # patches_flat: (batch, out_h, out_w, input_depth * k * k)
        # kernels_flat.T: (input_depth * k * k, depth)
        # Result: (batch, out_h, out_w, depth)
        output = self.patches_flat @ self.kernels_flat.T
        
        # step 5: transpose to (batch, depth, out_h, out_w) to match original output shape
        output = output.transpose(0, 3, 1, 2)
        
        # step 6: Add biases - shape (depth, 1, 1) broadcasts across batch, height, width
        output += self.biases[None, :, :, :]
        
        self.pre_activation_output = output
        self.output = self.ReLu(output)
        return self.output
    
    # def backward_prop(self, output, learning_rate=0.001):
    #     batch_size = self.input.shape[0]
    #     output_gradient = output * self.Derivative_ReLu(self.pre_activation_output)
    #     kernels_gradient = np.zeros(self.kernels_shape)
    #     input_gradient = np.zeros_like(self.input)
    
    #     for i in range(self.depth):
    #         for j in range(self.input_depth):
    #             correlations = [
    #                 signal.correlate2d(
    #                     self.input[b, j],
    #                     output_gradient[b, i],
    #                     mode="valid"
    #                 )
    #                 for b in range(batch_size)
    #             ]
    #             kernels_gradient[i, j] = np.mean(np.stack(correlations), axis=0)
    #             for b in range(batch_size):
                    
    #                 input_gradient[b, j] += signal.convolve2d(
    #                     output_gradient[b, i],
    #                     self.kernels[i, j],
    #                     mode="full"
    #                 )
    #     bias_gradient = np.sum(output_gradient, axis=(0, 2, 3)) / batch_size
    #     self.kernels -= learning_rate * kernels_gradient
    #     self.biases -= learning_rate * bias_gradient[:, None, None]
    #     return input_gradient
    
    def backward_prop(self, output, learning_rate=0.001):
        batch_size = self.input.shape[0]
        _, out_h, out_w = self.output_shape
        k = self.kernels_shape[2]
        
        # apply ReLu derivative
        output_gradient = output * self.Derivative_ReLu(self.pre_activation_output)
        # shape: (batch, depth, out_h, out_w)
        
        # --- kernel gradient ---
        # output_gradient: (batch, depth, out_h, out_w)
        # reshape it to (batch, out_h, out_w, depth) to align with patches flat
        og_reshaped = output_gradient.transpose(0, 2, 3, 1)
        # shape: (batch, out_h, out_w, depth)
        
        # flatten batch and spatial dimensions
        og_flat = og_reshaped.reshape(-1, self.depth)
        # shape: (batch * out_h * out_w, depth)
        
        pf_flat = self.patches_flat.reshape(-1, self.patches_flat.shape[-1])
        # shape: (batch * out_h * out_w, input_depth * k * k)
        
        # kernels_gradient shape: (depth, input_depth * k * k)
        # each kernel gets the dot product of its output_gradients with its input patches
        kernels_gradient = (og_flat.T @ pf_flat) / batch_size
        # reshape back to (depth, input_depth, k, k)
        kernels_gradient = kernels_gradient.reshape(self.kernels_shape)
        
        # --- input gradient ---
        # in the loop: input_gradient[i, j] += convolve2d(output_gradient[b, i], kernels[i, j], full)
        # full convolution = correlation with flipped kernel
        # this is the same as: input_grad_flat = output_gradient_flat @ kernels_flat (no transpose)
        
        # og_flat (batch * out_h * out_w, depth)
        # kernels_flat: (depth, input_depth * k * k)
        input_grad_patches = og_flat @ self.kernels_flat # (batch * out_h * out_w, input_depth * k * k)
        # reshape to (batch, out_h, out_w, input_depth, k, k)
        input_grad_patches = input_grad_patches.reshape(batch_size, out_h, out_w, self.input_depth, k, k)
        
        # now accumulate patch contributions back into the input gradient (col2im)
        input_gradient = np.zeros_like(self.input)
        for i in range(out_h):
            for j in range(out_w):
                input_gradient[:, :, i:i+k, j:j+k] += input_grad_patches[:, i, j, :, :, :]
        
        # --- bias gradient ---
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3)) / batch_size
        
        kernels_gradient = np.clip(kernels_gradient, -10.0, 10.0)
        bias_gradient = np.clip(bias_gradient, -10.0, 10.0)
        
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