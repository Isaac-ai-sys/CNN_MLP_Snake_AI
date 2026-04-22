from neural_network.nn import NN
from neural_network.train import Train


if __name__ == "__main__":
    nn = NN()
    nn.add_convolution_layer((3, 20, 20), 3, 3)
    nn.add_reshape_layer((3, 18, 18), (3 * 18 * 18, 1))
    nn.add_dense_layer(3 * 18 * 18, 256)
    nn.add_dense_layer(256, 32)
    nn.add_dense_layer(32, 10)
    nn.add_dense_layer(10, 4)
    # nn.load()
    t = Train(nn)
    
    while True:
        for i in range(10):
            t.train()
        nn.save()