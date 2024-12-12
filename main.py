from dataGenerator import DataGenerator
from network import Network
from config import read_config
from activations import *


def main():
    # Read config file
    run_options_dict, data_dict, network_dict, layers_dict = read_config()

    # Generate dataset according to config file
    data_gen = DataGenerator(data_dict)
    trainX, trainY, valX, valY, testX, testY = data_gen.generate_dataset()

    # Create the network
    neural_net = Network(network_dict, layers_dict)

    # Running options
    if run_options_dict['show_dataset']:
        data_gen.show_dataset(trainX)
    if run_options_dict['train_network']:
        neural_net.train(trainX, trainY, valX, valY)
        if run_options_dict['plot_learning']:
            neural_net.test(testX, testY)
            neural_net.plot_learning()
    # If specified in config: show hinton or verbose output for a specific layer
    neural_net.show_hintons()
    neural_net.verbose_output()



if __name__ == '__main__':
    np.random.seed(1)
    main()

























