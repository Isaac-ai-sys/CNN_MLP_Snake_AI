try:
    import cupy as xp
except:
    import numpy as xp

class Max_Pool:
    def __init__(self, size=2):
        self.pool_size = size

    def forward_prop(self, input):
        """
        input shape:
        (batch, depth, height, width)
        """

        self.input = input

        batch, depth, h, w = input.shape
        p = self.pool_size

        # ensure divisible by pool size
        h_trim = (h // p) * p
        w_trim = (w // p) * p

        input = input[:, :, :h_trim, :w_trim]

        out_h = h_trim // p
        out_w = w_trim // p

        # reshape into pooling windows
        x = input.reshape(
            batch,
            depth,
            out_h,
            p,
            out_w,
            p
        )

        # rearrange dimensions so pooling windows are grouped
        x = x.transpose(0, 1, 2, 4, 3, 5)

        self.x_reshaped = x

        # max pool
        output = x.max(axis=(4, 5))

        return output

    # max_pool.py backward_prop — replace with average pooling gradient
    def backward_prop(self, dout, learning_rate=0.001):
        p = self.pool_size
        # distribute gradient evenly across the pool window
        dout_expanded = dout[:, :, :, :, None, None]
        max_vals = self.x_reshaped.max(axis=(4, 5), keepdims=True)
        mask = (self.x_reshaped == max_vals)
        dx = mask * dout_expanded / mask.sum(axis=(4,5), keepdims=True).clip(min=1)
        dx = dx.transpose(0, 1, 2, 4, 3, 5)
        return dx.reshape(self.input.shape)
    
    def save(self, filename):
        return
    
    def load(self, filename):
        return