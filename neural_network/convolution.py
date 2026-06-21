USE_GPU = True

if USE_GPU:
    import cupy as xp
else:
    import numpy as xp


class Convolution:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape

        self.depth = int(depth)
        self.input_depth = int(input_depth)

        self.input_shape = tuple(map(int, input_shape))

        self.output_shape = (
            int(depth),
            int(input_height - kernel_size + 1),
            int(input_width - kernel_size + 1)
        )

        self.kernels_shape = (
            int(depth),
            int(input_depth),
            int(kernel_size),
            int(kernel_size)
        )

        scale = xp.sqrt(
            2 / (kernel_size * kernel_size * input_depth)
        )

        self.kernels = (
            xp.random.randn(*self.kernels_shape) * scale
        ).astype(xp.float32)

        self.biases = xp.zeros(
            (depth, 1, 1),
            dtype=xp.float32
        )
        # Adam optimizer state
        self.adam_m_k = xp.zeros_like(self.kernels)
        self.adam_v_k = xp.zeros_like(self.kernels)
        self.adam_m_b = xp.zeros_like(self.biases)
        self.adam_v_b = xp.zeros_like(self.biases)
        self.adam_t = 0

    def forward_prop(self, input):

        if input.ndim == 3:
            input = input[None, :]

        if USE_GPU and not isinstance(input, xp.ndarray):
            input = xp.asarray(input)

        self.input = input

        batch_size, _, h, w = input.shape

        k = int(self.kernels_shape[2])

        out_h = int(h - k + 1)
        out_w = int(w - k + 1)

        # ---------------------------------------
        # Sliding window extraction
        # ---------------------------------------

        shape = tuple(map(int, (
            batch_size,
            self.input_depth,
            out_h,
            out_w,
            k,
            k
        )))

        strides = tuple(map(int, (
            input.strides[0],
            input.strides[1],
            input.strides[2],
            input.strides[3],
            input.strides[2],
            input.strides[3]
        )))

        patches = xp.lib.stride_tricks.as_strided(
            input,
            shape=shape,
            strides=strides
        )

        # (batch, in_depth, out_h, out_w, k, k)
        # ->
        # (batch, out_h, out_w, in_depth, k, k)

        patches = patches.transpose(
            0,
            2,
            3,
            1,
            4,
            5
        )

        # ---------------------------------------
        # Flatten patches
        # ---------------------------------------

        self.patches_flat = patches.reshape(
            batch_size,
            out_h,
            out_w,
            -1
        )

        # ---------------------------------------
        # Flatten kernels
        # ---------------------------------------

        self.kernels_flat = self.kernels.reshape(
            self.depth,
            -1
        )

        # ---------------------------------------
        # Convolution via GEMM
        # ---------------------------------------

        output = (
            self.patches_flat
            @ self.kernels_flat.T
        )

        # (batch, out_h, out_w, depth)
        # ->
        # (batch, depth, out_h, out_w)

        output = output.transpose(
            0,
            3,
            1,
            2
        )

        # ---------------------------------------
        # Add bias
        # ---------------------------------------

        output += self.biases[None]

        self.pre_activation_output = output

        self.output = self.ReLu(output)

        return self.output

    def backward_prop(
        self,
        output,
        learning_rate=0.001,
        max_grad_norm=0.5,
        optimizer='sgd'
    ):

        if USE_GPU and not isinstance(output, xp.ndarray):
            output = xp.asarray(output)

        batch_size = int(self.input.shape[0])

        k = int(self.kernels.shape[2])

        out_h = int(output.shape[2])
        out_w = int(output.shape[3])

        # ---------------------------------------
        # ReLU derivative
        # ---------------------------------------

        output_gradient = (
            output
            * self.Derivative_ReLu(
                self.pre_activation_output
            )
        )

        # ---------------------------------------
        # Kernel gradients
        # ---------------------------------------

        og_reshaped = output_gradient.transpose(
            0,
            2,
            3,
            1
        )

        og_flat = og_reshaped.reshape(
            -1,
            self.depth
        )

        pf_flat = self.patches_flat.reshape(
            -1,
            self.patches_flat.shape[-1]
        )

        kernels_gradient = (
            og_flat.T @ pf_flat
        ) / batch_size

        kernels_gradient = kernels_gradient.reshape(
            self.kernels_shape
        )

        # ---------------------------------------
        # Input gradient patches
        # ---------------------------------------

        input_grad_patches = (
            og_flat @ self.kernels_flat
        )

        input_grad_patches = input_grad_patches.reshape(
            batch_size,
            out_h,
            out_w,
            self.input_depth,
            k,
            k
        )

        # ---------------------------------------
        # Input gradient via full convolution
        # (replaces broken as_strided col2im)
        # ---------------------------------------

        input_h = int(self.input.shape[2])
        input_w = int(self.input.shape[3])

        # Pad output_gradient by (k-1) on all sides
        pad = k - 1
        og_padded = xp.pad(
            output_gradient,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode='constant'
        )

        shape = (
            batch_size,
            self.depth,
            input_h,
            input_w,
            k,
            k
        )

        strides = (
            og_padded.strides[0],
            og_padded.strides[1],
            og_padded.strides[2],
            og_padded.strides[3],
            og_padded.strides[2],
            og_padded.strides[3],
        )

        grad_patches = xp.lib.stride_tricks.as_strided(
            og_padded,
            shape=tuple(map(int, shape)),
            strides=tuple(map(int, strides))
        )

        # (batch, depth, in_h, in_w, k, k)
        # -> (batch, in_h, in_w, depth, k, k)
        grad_patches = grad_patches.transpose(0, 2, 3, 1, 4, 5)
        grad_patches_flat = grad_patches.reshape(batch_size, input_h, input_w, -1)

        # Flip kernels 180° for transposed convolution
        # kernels: (depth, in_depth, k, k) -> flip spatial dims
        kernels_flipped = self.kernels[:, :, ::-1, ::-1]
        # (depth, in_depth*k*k)
        # We need (in_depth*k*k, depth) to map grad_patches -> input_gradient
        # But we want output (batch, in_h, in_w, in_depth)
        # grad_patches_flat: (batch, in_h, in_w, depth*k*k)
        # kernels_flipped rearranged: (depth*k*k, in_depth)
        kernels_for_dx = kernels_flipped.transpose(1, 0, 2, 3).reshape(self.input_depth, -1).T
        # kernels_for_dx: (depth*k*k, in_depth)

        input_gradient = grad_patches_flat @ kernels_for_dx
        # (batch, in_h, in_w, in_depth)
        # -> (batch, in_depth, in_h, in_w)
        input_gradient = input_gradient.transpose(0, 3, 1, 2)

        # ---------------------------------------
        # Bias gradients
        # ---------------------------------------

        bias_gradient = (
            xp.sum(
                output_gradient,
                axis=(0, 2, 3)
            )
            / batch_size
        )

        # ---------------------------------------
        # Gradient clipping
        # ---------------------------------------

        grad_norm = xp.sqrt(
            xp.sum(kernels_gradient ** 2) +
            xp.sum(bias_gradient ** 2)
        )

        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-8)

            kernels_gradient *= scale
            bias_gradient *= scale

        # ---------------------------------------
        # SGD update
        # ---------------------------------------

        if optimizer == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            self.adam_t += 1

            self.adam_m_k = beta1 * self.adam_m_k + (1 - beta1) * kernels_gradient
            self.adam_v_k = beta2 * self.adam_v_k + (1 - beta2) * (kernels_gradient ** 2)
            m_hat_k = self.adam_m_k / (1 - beta1 ** self.adam_t)
            v_hat_k = self.adam_v_k / (1 - beta2 ** self.adam_t)
            self.kernels -= learning_rate * m_hat_k / (xp.sqrt(v_hat_k) + eps)

            # Expand bias gradient to (depth,1,1) so Adam state shapes remain consistent
            bias_grad_exp = bias_gradient[:, None, None]
            self.adam_m_b = beta1 * self.adam_m_b + (1 - beta1) * bias_grad_exp
            self.adam_v_b = beta2 * self.adam_v_b + (1 - beta2) * (bias_grad_exp ** 2)
            m_hat_b = self.adam_m_b / (1 - beta1 ** self.adam_t)
            v_hat_b = self.adam_v_b / (1 - beta2 ** self.adam_t)
            self.biases -= learning_rate * m_hat_b / (xp.sqrt(v_hat_b) + eps)
        else:
            self.kernels -= (
                learning_rate
                * kernels_gradient
            )

            self.biases -= (
                learning_rate
                * bias_gradient[:, None, None]
            )

        return input_gradient

    # Fix: match Dense's leaky ReLU
    def ReLu(self, z):
        return xp.where(z > 0, z, 0.01 * z)

    def Derivative_ReLu(self, z):
        return xp.where(z > 0, 1.0, 0.01).astype(xp.float32)

    def save(self, filename):

        if USE_GPU:
            kernels = xp.asnumpy(self.kernels)
            biases = xp.asnumpy(self.biases)
        else:
            kernels = self.kernels
            biases = self.biases

        xp.savez(
            filename,
            kernels=kernels,
            biases=biases,
            input_depth=self.input_depth,
            depth=self.depth,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            kernels_shape=self.kernels_shape
        )

    def load(self, filename):

        data = xp.load(
            filename,
            allow_pickle=True
        )

        self.kernels = xp.asarray(
            data["kernels"]
        ).astype(xp.float32)

        self.biases = xp.asarray(
            data["biases"]
        ).astype(xp.float32)

        self.input_depth = int(
            data["input_depth"]
        )

        self.depth = int(
            data["depth"]
        )

        self.input_shape = tuple(
            map(int, data["input_shape"])
        )

        self.output_shape = tuple(
            map(int, data["output_shape"])
        )

        self.kernels_shape = tuple(
            map(int, data["kernels_shape"])
        )

        if self.kernels.shape != self.kernels_shape:
            raise ValueError(
                f"Loaded convolution kernels shape {self.kernels.shape} does not match expected {self.kernels_shape}"
            )

        if self.biases.shape != (self.depth, 1, 1):
            raise ValueError(
                f"Loaded convolution biases shape {self.biases.shape} does not match expected {(self.depth, 1, 1)}"
            )
