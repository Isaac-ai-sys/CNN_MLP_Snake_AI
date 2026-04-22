from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    nn = NN()
    nn.add_convolution_layer((3, 20, 20), 3, 3)
    nn.add_reshape_layer((3, 18, 18), (3 * 18 * 18, 1))
    nn.add_dense_layer(256, 3 * 18 * 18 + 4 + 1)
    nn.add_dense_layer(32, 256)
    nn.add_dense_layer(10, 32)
    nn.add_dense_layer(4, 10)
    # nn.load()
    t = Train(nn)
    
    while True:
        t.train()
        nn.save()