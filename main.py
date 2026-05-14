from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    BOARD_SIZE = 20
    KERNEL_SIZE = 3
    CONV_DEPTH = 3

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
    
    # nn.add_dense_layer(32, dense_input)
    # nn.add_dense_layer(4, 32)
    #nn.load()
    t = Train(nn, board_size=BOARD_SIZE)
    epoch_max = 0
    entropy_min = 2
    while True:
        epoch_avg, entropy_avg = t.train(epochs=10, episodes=16)
        
        if(epoch_avg > epoch_max):
            epoch_max = epoch_avg
            print(f"epoch_avg: {epoch_avg:.3f} ****** New Max ******")
        else:
            print(f"epoch_avg: {epoch_avg:.3f}")
        
        # if(entropy_avg < entropy_min):
        #     entropy_min = entropy_avg
        #     print(f"entropy_avg: {entropy_avg:.3f} ****** New Min ******")
        # else:
        #     print(f"entropy_avg: {entropy_avg:.3f}")
        
        nn.save()