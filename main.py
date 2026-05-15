from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    BOARD_SIZE = 20
    KERNEL_SIZE = 3
    CONV_DEPTH = 8

    conv_out_h = BOARD_SIZE - KERNEL_SIZE + 1
    conv_out_w = BOARD_SIZE - KERNEL_SIZE + 1
    flat_size = CONV_DEPTH * conv_out_h * conv_out_w  # conv output
    dense_input = flat_size + 4 + 1                   # + direction (4) + length (1)

    feature_layers = []
    actor_layers = []
    critic_layers = []
    
    nn = NN()
    # create feature network
    feature_layers.append(nn.create_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH))
    feature_layers.append(nn.create_reshape_layer((CONV_DEPTH, conv_out_h, conv_out_w), (flat_size, 1)))
    feature_layers.append(nn.create_dense_layer(128, dense_input))
    nn.feature_layers = feature_layers
    
    #create actor network
    actor_layers.append(nn.create_dense_layer(32, 128))
    actor_layers.append(nn.create_dense_layer(4, 32))
    nn.actor_layers = actor_layers
    
    #create critic network
    critic_layers.append(nn.create_dense_layer(32, 128))
    critic_layers.append(nn.create_dense_layer(1, 32))
    nn.critic_layers = critic_layers

    #nn.load()
    t = Train(nn, board_size=BOARD_SIZE)
    epoch_max = 0
    entropy_min = 2
    while True:
        epoch_avg = t.train(epochs=5, episodes=100)
        
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