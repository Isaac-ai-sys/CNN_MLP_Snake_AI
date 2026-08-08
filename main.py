from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    BOARD_SIZE = 20
    KERNEL_SIZE = 3
    CONV_DEPTH = 8
    POOL_SIZE = 2

    conv_out = BOARD_SIZE

    feature_layers = []
    actor_layers = []
    critic_layers = []
    value_layers = []
    
    nn = NN()
    # create feature network
    feature_layers.append(nn.create_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH, padding=1)) # output is 20x20x8 : 3x3x3x8 + 8 = 224 parameters
    feature_layers.append(nn.create_convolution_layer((CONV_DEPTH, conv_out, conv_out), KERNEL_SIZE, CONV_DEPTH * 2, padding=1)) # output is 20x20x16 : 16x8x3x3 + 16 = 1,168 parameters
    CONV_DEPTH = CONV_DEPTH * 2
    feature_layers.append(nn.create_convolution_layer((CONV_DEPTH, conv_out, conv_out), KERNEL_SIZE, CONV_DEPTH * 2, padding=1)) # output is 20x20x32 : 32×16×3×3 + 32 = 4,640 parameters
    CONV_DEPTH = CONV_DEPTH * 2
    flat_size = CONV_DEPTH * conv_out * conv_out
    dense_input = flat_size + 8
    feature_layers.append(nn.create_reshape_layer((CONV_DEPTH, conv_out, conv_out), (flat_size, 1)))
    feature_layers.append(nn.create_dense_layer(128, dense_input)) # (20x20x32 + 8 + 1) x 128 = 1,639,552 parameters
    nn.feature_layers = feature_layers
    
    #create actor network
    actor_layers.append(nn.create_dense_layer(128, 128)) # 128x128 + 128 = 16,512 parameters
    actor_layers.append(nn.create_dense_layer(64, 128)) # 64x128 + 64 = 8,256 parameters
    actor_layers.append(nn.create_dense_layer(32, 64)) # 64x32 + 32 = 2,080 parameters
    actor_layers.append(nn.create_dense_layer(16, 32)) #  16x32 + 16 = 528 parameters
    actor_layers.append(nn.create_dense_layer(4, 16)) # 16x4 + 4 = 68 parameters
    nn.actor_layers = actor_layers
    
    #create critic network
    critic_layers.append(nn.create_dense_layer(32, 128)) # 32x128 + 32 = 4,128 parameters
    critic_layers.append(nn.create_dense_layer(1, 32)) # 32x1 + 1 = 33 parameters
    nn.critic_layers = critic_layers

    nn.load()
    t = Train(nn, board_size=BOARD_SIZE, num_envs=512)
    max_avg = 0
    entropy = 1.0
    avg_length = 0
    entropy_coef = 0.1
    actor_learning_rate = 0.003
    critic_learning_rate = 0.005
    while True:
        avg_length, max_length = t.test(avg_length)
        if(avg_length > max_avg and entropy > 0.2):
            max_avg = avg_length
            print(f"epoch_avg: {avg_length:.3f} ****** New Max ******")
            # print(f"entropy: {entropy:.3f}")
            nn.save()
        else:
            print(f"epoch_avg: {avg_length:.3f}")
        returns_avg, entropy, entropy_coef, actor_learning_rate, critic_learning_rate = t.train(
            verbose=False, 
            entropy_coef=entropy_coef, 
            actor_learning_rate=actor_learning_rate, 
            critic_learning_rate=critic_learning_rate
        )