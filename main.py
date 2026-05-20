from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    BOARD_SIZE = 20
    KERNEL_SIZE = 3
    CONV_DEPTH = 8
    POOL_SIZE = 2

    conv1_out = BOARD_SIZE - KERNEL_SIZE + 1
    pool_out = conv1_out // POOL_SIZE
    conv2_out = pool_out - KERNEL_SIZE + 1

    flat_size = (CONV_DEPTH * 2) * conv2_out * conv2_out
    dense_input = flat_size + 11

    feature_layers = []
    actor_layers = []
    critic_layers = []
    
    nn = NN()
    # create feature network
    feature_layers.append(nn.create_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH)) # output is 18x18x8
    feature_layers.append(nn.create_max_pool_layer(2)) # output is 9x9x8
    feature_layers.append(nn.create_convolution_layer((CONV_DEPTH, conv1_out, conv1_out), KERNEL_SIZE, CONV_DEPTH * 2)) # output is 7x7x16
    feature_layers.append(nn.create_reshape_layer((CONV_DEPTH * 2, conv2_out, conv2_out), (flat_size, 1)))
    feature_layers.append(nn.create_dense_layer(128, dense_input)) # (7x7x16 + 11) x 128 = 101,760 parameters
    nn.feature_layers = feature_layers
    
    #create actor network
    actor_layers.append(nn.create_dense_layer(32, 128)) # 128x32 = 4096 parameters
    actor_layers.append(nn.create_dense_layer(4, 32)) # 32x4 = 128 parameters
    nn.actor_layers = actor_layers
    
    #create critic network
    critic_layers.append(nn.create_dense_layer(32, 128)) # 128x32 = 4096 parameters
    critic_layers.append(nn.create_dense_layer(1, 32)) # 32x1 = parameters
    nn.critic_layers = critic_layers

    nn.load()
    t = Train(nn, board_size=BOARD_SIZE)
    epoch_max = 0
    entropy_min = 2
    while True:
        epoch_avg = t.train(epochs=5, episodes=64)
        
        if(epoch_avg > epoch_max):
            epoch_max = epoch_avg
            print(f"epoch_avg: {epoch_avg:.3f} ****** New Max ******")
            nn.save()
        else:
            print(f"epoch_avg: {epoch_avg:.3f}")
        
        # if(entropy_avg < entropy_min):
        #     entropy_min = entropy_avg
        #     print(f"entropy_avg: {entropy_avg:.3f} ****** New Min ******")
        # else:
        #     print(f"entropy_avg: {entropy_avg:.3f}")