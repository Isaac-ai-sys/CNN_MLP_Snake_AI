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
    dense_input = flat_size + 8

    feature_layers = []
    actor_layers = []
    critic_layers = []
    value_layers = []
    
    nn = NN()
    # create feature network
    feature_layers.append(nn.create_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH)) # output is 18x18x8
    feature_layers.append(nn.create_max_pool_layer(2)) # output is 9x9x8
    feature_layers.append(nn.create_convolution_layer((CONV_DEPTH, pool_out, pool_out), KERNEL_SIZE, CONV_DEPTH * 2)) # output is 7x7x16
    feature_layers.append(nn.create_reshape_layer((CONV_DEPTH * 2, conv2_out, conv2_out), (flat_size, 1)))
    feature_layers.append(nn.create_dense_layer(64, dense_input)) # (7x7x16 + 12) x 64 = 50,944 parameters
    nn.feature_layers = feature_layers
    
    #create actor network
    actor_layers.append(nn.create_dense_layer(32, 64)) # 64x32 = 2048 parameters
    actor_layers.append(nn.create_dense_layer(16, 32)) #  16x32 = 512 parameters
    actor_layers.append(nn.create_dense_layer(4, 16)) # 16x4 = 64 parameters
    nn.actor_layers = actor_layers
    
    #create critic network
    critic_layers.append(nn.create_dense_layer(32, 64)) # 64x32 = 2048 parameters
    critic_layers.append(nn.create_dense_layer(1, 32)) # 32x1 = parameters
    nn.critic_layers = critic_layers
    
    #create value network
    value_layers.append(nn.create_dense_layer(32, 64)) # 64x32 = 2048 parameters
    value_layers.append(nn.create_dense_layer(1, 32)) # 32x1 = parameters
    nn.value_layers = value_layers

    nn.load()
    t = Train(nn, board_size=BOARD_SIZE, num_envs=128)
    max_avg = 0
    entropy = 1.0
    avg_length = 60
    while True:
        avg_length, max_length = t.test(avg_length)
        if(avg_length > max_avg and entropy > 0.2):
            max_avg = avg_length
            print(f"epoch_avg: {avg_length:.3f} ****** New Max ******")
            # print(f"entropy: {entropy:.3f}")
            nn.save()
        else:
            print(f"epoch_avg: {avg_length:.3f}")
        returns_avg, entropy = t.train(verbose=False)
        