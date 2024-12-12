import numpy as np
from denseLayer import FullyConnectedLayer
from convLayer2D import ConvLayer2D
from convLayer1D import ConvLayer1D
import matplotlib.pyplot as plt


class Network:
    def __init__(self, network_dict, layers_dict):
        self.lr = network_dict['learning_rate']
        self.loss = network_dict['loss_function']
        self.softmax = network_dict['softmax']
        self.do_smooth_labels = network_dict['smooth_labels']
        self.input_size = network_dict['input_size']
        self.epochs = network_dict['epochs']
        self.minibatch_size = network_dict['minibatch_size']

        if network_dict['loss_function'] == 'mse':
            self.loss = mean_square_error
        elif network_dict['loss_function'] == 'cross_entropy':
            self.loss = cross_entropy

        self.layers = []
        self.init_layers(layers_dict)
        self.training_error = []
        self.training_count = []
        self.test_error = []
        self.test_count = []
        self.val_error = []
        self.val_count = []

    def init_layers(self, layer_configs):
        incoming_channels = 1
        layer_index = 1
        for key in layer_configs.keys():
            if layer_configs[key]['type'] == 'conv':
                if type(layer_configs[key]['filter_size']) == tuple:
                    layer = ConvLayer2D(incoming_channels,
                                        layer_configs[key]['number_of_filters'],
                                        layer_configs[key]['filter_size'],
                                        layer_configs[key]['mode'],
                                        layer_configs[key]['stride'],
                                        layer_configs[key]['weight_init'],
                                        layer_configs[key]['activation'],
                                        layer_index,
                                        layer_configs[key]['show_hinton'],
                                        layer_configs[key]['verbose'])
                else:
                    layer = ConvLayer1D(incoming_channels,
                                        layer_configs[key]['number_of_filters'],
                                        layer_configs[key]['filter_size'],
                                        layer_configs[key]['mode'],
                                        layer_configs[key]['stride'],
                                        layer_configs[key]['weight_init'],
                                        layer_configs[key]['activation'],
                                        layer_index,
                                        layer_configs[key]['show_hinton'],
                                        layer_configs[key]['verbose'])
                incoming_channels = layer_configs[key]['number_of_filters']
            elif layer_configs[key]['type'] == 'dense':
                layer = FullyConnectedLayer(layer_configs[key]['neurons'],
                                            layer_configs[key]['weight_init'],
                                            layer_configs[key]['activation'],
                                            layer_index,
                                            layer_configs[key]['show_hinton'],
                                            layer_configs[key]['verbose'])
                incoming_channels = 1
            layer_index += 1
            self.layers.append(layer)

    def train(self, trainX, trainY, valX, valY):
        val_size = len(valX)
        val_count = 0

        if self.do_smooth_labels:
            trainY = self.smooth_labels(trainY)
            valY = self.smooth_labels(valY)

        epoch_count = 1
        train_count = 1
        for epoch in range(self.epochs):
            print('===============================')
            print(f'Epoch : {epoch_count}/{self.epochs}')
            for i in range(0, len(trainX), self.minibatch_size):
                minibatch = trainX[i:i+self.minibatch_size]
                label_batch = trainY[i:i+self.minibatch_size]

                prediction = self.forward_pass(minibatch)
                self.backward_pass(prediction, label_batch)
                self.update_weights()

                error = self.loss(prediction, label_batch)
                self.training_error.append(error.mean(axis=0))
                self.training_count.append(train_count)
                train_count += self.minibatch_size
                if i < val_size:
                    val_predY = self.forward_pass(valX[i:i+self.minibatch_size])
                    val_error = self.loss(val_predY, valY[i:i+self.minibatch_size])
                    self.val_error.append(val_error.mean(axis=0))
                    self.val_count.append(val_count)
                    val_count += self.minibatch_size
            epoch_count += 1

    def test(self, testX, testY):
        test_count = self.training_count[-1]
        correct = 0
        wrong = 0
        for i in range(len(testX)):
            testcase = testX[i:i+self.minibatch_size]
            testlabel = testY[i:i+self.minibatch_size]
            predY = self.forward_pass(testcase)
            loss = self.loss(predY, testlabel)
            self.test_error.append(loss.mean(axis=0))
            test_count += 1
            self.test_count.append(test_count)

            for j in range(len(testlabel)):
                target_i = np.argmax(testlabel[j])
                pred_i = np.argmax(predY[j])
                if target_i == pred_i:
                    correct += 1
                else:
                    wrong += 1
        if correct + wrong > 0:
            print(f'Test: {100*round(correct/(correct+wrong), 2)} % correct')

    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)

        if self.softmax:
            return softmax(x)
        else:
            return x

    def backward_pass(self, predY, caseY):
        if self.softmax:
            if self.loss == cross_entropy:
                J_L_Z = cross_entropy(predY, caseY, get_derivative=True, softmax=True)
            else:
                J_L = self.loss(predY, caseY, get_derivative=True)
                J_soft = softmax(predY, get_derivative=True)
                # Instead of taking dot product of each minibatch case
                J_L_Z = np.einsum('ijk,ik->ij', J_soft, J_L)
        else:
            J_L_Z = self.loss(predY, caseY, get_derivative=True)

        for layer in reversed(self.layers):
            J_L_Y = layer.backward_pass(J_L_Z)
            J_L_Z = J_L_Y

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.lr)


    def smooth_labels(self, hard_target):
        soft_target = hard_target * 0.95 + 0.5 * (1 - hard_target)
        return soft_target

    def plot_learning(self):
        plt.plot(self.training_count,
                 self.training_error,
                 label='Train')
        plt.plot(self.val_count,
                 self.val_error,
                 label='Val')
        plt.plot(self.test_count,
                 self.test_error,
                 label='Test')
        plt.xlabel('Minibatches x minibatch size')
        plt.ylabel('Error')
        plt.title('Learning plot')
        plt.legend()
        plt.show()

    def show_hintons(self):
        for layer in self.layers:
            layer.show_hinton()

    def verbose_output(self):
        for layer in self.layers:
            layer.show_verbose()


def mean_square_error(predicted, target, get_derivative=False):
    if get_derivative:
        return 2/target.shape[1] * (predicted - target)
    else:
        return np.mean((predicted - target)**2)


def cross_entropy(predicted, target, epsilon=1e-9, get_derivative=False, softmax=False):
    if get_derivative and softmax:
        return predicted - target
    elif get_derivative:
        return -target * 1/(predicted+epsilon)
    else:
        return -np.sum(target * np.log(predicted + epsilon), axis=1, keepdims=True)


def softmax(z, get_derivative=False):
    if get_derivative:
        # Calculating the Jacobian tensor for softmax input
        t1 = np.einsum('ij,ik->ijk', z, z) # Outer product with itself
        t2 = np.einsum('ij,jk->ijk', z, np.eye(z.shape[1], z.shape[1])) # Identity matrix
        J_soft = t2 - t1
        return J_soft
    else:
        expZ = np.exp(z - np.amax(z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)






