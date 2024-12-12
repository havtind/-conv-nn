from activations import *

class FullyConnectedLayer:
    def __init__(self, num_nodes, weight_init, act,
                 layer_index=0, show_hinton=0, verbose=0):
        self.num_nodes = num_nodes
        self.act = eval(act)
        self.layer_index = layer_index
        self.do_show_hinton = show_hinton
        self.verbose = verbose
        if type(weight_init) == tuple:
            self.wr = weight_init
            self.init_weights = self.init_range_weights
        else:
            self.init_weights = self.init_glorot_weights
        self.bias = np.zeros(num_nodes)

        self.output = None
        self.input = None
        self.input_size = None
        self.weights = None
        self.first_iteration = True

    def init_range_weights(self):
        self.weights = np.random.uniform(self.wr[0], self.wr[1], (self.input_size, self.num_nodes))

    def init_glorot_weights(self):
        sd = np.sqrt(6.0 / (self.input_size + self.num_nodes))
        self.weights = np.zeros((self.input_size, self.num_nodes))
        for i in range(self.input_size):
            for j in range(self.num_nodes):
                x = np.float32(np.random.uniform(-sd, sd))
                self.weights[i, j] = x

    def update_weights(self, lr=0.01):
        self.weights -= lr * self.J_L_W
        self.bias -= lr * self.J_L_B

    def forward_pass(self, _input):
        # Make sure the incoming activation has dimensions (minibatch x element_size)
        if np.ndim(_input) == 4: # from 2D conv
            _input = np.reshape(_input, (_input.shape[0], _input.shape[1] * _input.shape[2] * _input.shape[3]))
        elif np.ndim(_input) == 3: # from 1D conv
            _input = np.reshape(_input, (_input.shape[0], _input.shape[1] * _input.shape[2]))

        self.input_size = _input.shape[1]
        if self.first_iteration:
            self.init_weights()
            self.first_iteration = False

        self.input = _input
        self.dotprod = np.einsum('mi, ij->mj', _input, self.weights)
        self.output = self.act(self.dotprod)
        return self.output

    def backward_pass(self, J_L_Z):
        if np.ndim(J_L_Z) == 3: # If from a 1D layer
            J_L_Z = np.reshape(J_L_Z, (J_L_Z.shape[0], J_L_Z.shape[1] * J_L_Z.shape[2]))

        J_Z_S = self.act(self.dotprod, get_derivative=True)
        J_L_S = J_L_Z * J_Z_S

        # Transform input into (m x n_input x 1) and gradient into (m x 1 x n_output)
        # Take the matrix product for the two last dimensions.
        J_L_W = np.matmul(self.input[:, :, None], J_L_S[:, None, :])

        # Taking mean over minibatch dim
        self.J_L_W = J_L_W.mean(axis=0)
        self.J_L_B = J_L_S.mean(axis=0)

        J_L_Y = np.einsum('mj, ij-> mi', J_L_S, self.weights)

        return J_L_Y

    def show_verbose(self):
        if self.verbose:
            print('---------------------------')
            print(f'L{self.layer_index}: DENSE ')
            print(f'Initial input shape: { self.initial_input.shape}')
            print(f'Adapted input shape: {self.input.shape}')
            print(f'Weight shape: {self.weights.shape}')
            print(f'Output shape: {self.output.shape}')
            print('***')
            print(f'Input values:\n{self.input}')
            print(f'Output values:\n{self.output}')
            print('***')

    def show_hinton(self):
        if self.do_show_hinton:
            m = self.weights
            hinton(m)
            plt.title(f'Layer {self.layer_index}: Dense\nWeight shape: {self.weights.shape}')
            plt.show()
















