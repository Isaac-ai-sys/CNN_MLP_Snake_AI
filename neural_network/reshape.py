try:
    import cupy as np
except:
    import numpy as np
import math


class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_prop(
        self,
        input,
        direction,
        length,
        dx_food,
        dy_food,
        running
    ):

        # ensure CuPy arrays
        input = np.asarray(input)
        direction = np.asarray(direction)

        batch_size = input.shape[0]

        flat_input = input.reshape(batch_size, -1)
        flat_direction = direction.reshape(batch_size, -1)

        extras_list = []
        for value in (
            length,
            dx_food,
            dy_food,
            running
        ):
            value_arr = np.asarray(value, dtype=np.float32)
            if value_arr.ndim == 0:
                value_arr = np.full((batch_size, 1), value_arr, dtype=np.float32)
            elif value_arr.ndim == 1:
                if value_arr.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected batch length {batch_size} for extra feature, got {value_arr.shape}"
                    )
                value_arr = value_arr.reshape(batch_size, 1)
            else:
                value_arr = value_arr.reshape(batch_size, -1)
            extras_list.append(value_arr)

        extras = np.concatenate(extras_list, axis=1)

        return np.concatenate(
            [flat_input, flat_direction, extras],
            axis=1
        )

    def backward_prop(self, output, learning_rate=0.001):
        output = np.asarray(output)
        return output[:, :math.prod(self.input_shape)].reshape(
            (-1, *self.input_shape)
        )
    
    def save(self, filename):
        return
    
    def load(self, filename):
        return