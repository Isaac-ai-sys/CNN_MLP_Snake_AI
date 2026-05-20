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

    def backward_prop(self, dout, learning_rate=0.001):
        """
        dout shape:
        (batch, depth, out_h, out_w)
        """

        # identify max locations
        max_vals = self.x_reshaped.max(
            axis=(4, 5),
            keepdims=True
        )

        mask = (self.x_reshaped == max_vals)

        # expand upstream gradient
        dout = dout[:, :, :, :, None, None]

        # send gradient only to max positions
        dx = mask * dout

        # restore original arrangement
        dx = dx.transpose(0, 1, 2, 4, 3, 5)

        # reshape back to input shape
        dx = dx.reshape(self.input.shape)

        return dx
    
    def save(self, filename):
        return
    
    def load(self, filename):
        return