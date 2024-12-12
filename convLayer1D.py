import math
from activations import *


class ConvLayer1D:
    def __init__(self, num_input_channels, num_filters, filter_length,
                 mode, stride, init_wr, act,
                 layer_index=0, show_hinton=0, verbose=0):
        self.input_channels = num_input_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.mode = mode
        self.stride = stride
        self.verbose = verbose

        self.filters = np.random.uniform(init_wr[0], init_wr[1], (num_filters, num_input_channels, filter_length))
        self.act = eval(act)

        self.do_show_hinton = show_hinton
        self.layer_index = layer_index
        self.first = True

    def forward_pass(self, _input):
        # Make sure the incoming activation has dimensions (minibatch x input_channels x element_size)
        if np.ndim(_input) == 4:
            _input = np.reshape(_input, (_input.shape[0], _input.shape[1], _input.shape[2] * _input.shape[3]))
        elif len(_input.shape) == 2:
            _input = np.reshape(_input, (_input.shape[0], 1, _input.shape[1]))
        self.input = _input

        padding = self.number_of_zeros_to_add(self.mode, self.filter_length)
        output_len = self.get_output_dim(_input.shape[2] + padding, self.filter_length, self.stride)
        _s_output = np.zeros((_input.shape[0], self.num_filters, output_len))

        left_pad = 0
        for mb in range(_input.shape[0]):
            padded_input, left_pad = self.add_padding1D(_input[mb], self.mode, self.filter_length)
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    temp_output = self.conv_parser1D(padded_input[ch], self.filters[f, ch], self.stride)
                    _s_output[mb, f] += temp_output

        _output = self.act(_s_output)
        self.output = _output
        self._s_outout = _s_output
        self.left_pad = left_pad
        return _output

    def backward_pass(self, J_L_Z):
        # Make sure the incoming loss gradient has dimensions (minibatch x num_filters x out_dim)
        if np.ndim(J_L_Z) == 2:
            J_L_Z = np.reshape(J_L_Z, (J_L_Z.shape[0], self.num_filters, self.output.shape[2]))

        filter_len = self.filter_length
        J_Y_sum = self.act(self._s_outout, get_derivative=True)
        J_L_Y_sum = J_L_Z * J_Y_sum

        _input_len = self.input.shape[2]

        dL_W = np.zeros((J_L_Z.shape[0], self.num_filters, self.input_channels, filter_len))
        dL_X = np.zeros((J_L_Z.shape[0], self.input_channels, _input_len))

        out_len = J_L_Y_sum.shape[2]

        # Iterates through minibatch, filters, incoming channels and vector dimension
        for mb in range(self.input.shape[0]):
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    k = 0
                    for xloc in range(-self.left_pad, _input_len, self.stride):
                        for j in range(0, filter_len, 1):
                            i = xloc + j
                            if 0 <= i < _input_len and k < out_len:
                                dL_W[mb, f, ch, j] += self.input[mb, ch, i] * J_L_Y_sum[mb, f, k]
                                dL_X[mb, ch, i] += self.filters[f, ch, j] * J_L_Y_sum[mb, f, k]
                        k += 1
        self.dL_W = dL_W.mean(axis=0)
        return dL_X

    def update_weights(self, lr):
        self.filters -= lr * self.dL_W

    def conv_parser1D(self, _input, kernel, stride):
        k_len = len(kernel)
        res_len = math.floor((len(_input)-k_len)/stride) + 1
        res = np.zeros(res_len)
        res_index = 0
        for i in range(0, len(_input), stride):
            slice = _input[i: i + k_len]
            if len(slice) == k_len:
                value = np.sum(slice * kernel)
                res[res_index] = value
            res_index += 1
        return res

    def add_padding1D(self, _input, mode, k_len):
        hor_pad = self.number_of_zeros_to_add(mode, k_len)
        left_pad = math.floor(hor_pad / 2)
        ver_input_len = _input.shape[0]
        hor_input_len = _input.shape[1]
        padded_input = np.zeros((ver_input_len, hor_input_len + hor_pad))
        padded_input[0:ver_input_len, left_pad:hor_input_len + left_pad] = _input
        return padded_input, left_pad


    def get_output_dim(self, padded_input_dim, k_dim, stride):
        return math.floor((padded_input_dim - k_dim) / stride) + 1

    def show_hinton(self):
        if self.do_show_hinton:
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    m = self.filters[f, ch]
                    plt.title(f'Layer {self.layer_index}: 1D Conv   Filter shape: {self.filters.shape}\n'
                              f'Filter {f+1}/{self.num_filters} in input channel {ch+1}/{self.input_channels}')
                    hinton(np.array([m]))
                    plt.show()

    def show_verbose(self):
        if self.verbose:
            print('---------------------------')
            print(f'L{self.layer_index}: 1D CONV ')
            print(f'Adapted input shape: {self.input.shape}')
            print(f'Shape of filters: {self.filters.shape}')
            print(f'Output shape: {self.output.shape}')
            print('***')
            print(f'Input values:\n{self.input}')
            print(f'Output values:\n{self.output}')
            print('***')

    def number_of_zeros_to_add(self, mode, k_len):
        padding = 0
        if mode == 'full':
            padding = 2*k_len-2
        elif mode == 'same':
            padding = k_len-1
        elif mode == 'valid':
            pass
        return padding









