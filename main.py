from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    BOARD_SIZE = 6
    KERNEL_SIZE = 3
    CONV_DEPTH = 16

    conv_out_h = BOARD_SIZE - KERNEL_SIZE + 1
    conv_out_w = BOARD_SIZE - KERNEL_SIZE + 1
    flat_size = CONV_DEPTH * conv_out_h * conv_out_w  # conv output
    dense_input = flat_size + 4 + 1                   # + direction (4) + length (1)

    nn = NN()
    nn.add_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH)
    nn.add_reshape_layer((CONV_DEPTH, conv_out_h, conv_out_w), (flat_size, 1))
    nn.add_dense_layer(256, dense_input)
    nn.add_dense_layer(64, 256)
    nn.add_dense_layer(16, 64)
    nn.add_dense_layer(4, 16)
    #nn.load()
    t = Train(nn, board_size=BOARD_SIZE)
    while True:
        t.train(episodes=512)
        nn.save()