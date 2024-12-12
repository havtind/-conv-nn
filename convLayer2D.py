import math
from activations import *


class ConvLayer2D:
    def __init__(self, input_channels, num_filters, filter_dim,
                 modes, strides, weight_init, act,
                 layer_index=0, show_hinton=0, verbose=0):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.layer_index = layer_index
        self.act = eval(act)
        self.do_show_hinton = show_hinton
        self.verbose = verbose

        if type(modes) == tuple:
            self.modes = modes
        else:
            self.modes = (modes, modes)
        if type(strides) == tuple:
            self.strides = strides
        else:
            self.strides = (strides, strides)

        self.filters = np.random.uniform(weight_init[0], weight_init[1], (num_filters, input_channels, filter_dim[0], filter_dim[1]))

    def forward_pass(self, _input):
        # The input will always have dimensions (minibatch x input_channels x element_dim0 x element_dim1)
        self.input = _input

        # Get amount of padding in each dimension
        ver_pad = self.number_of_zeros_to_add(self.modes[0], self.filter_dim[0])
        hor_pad = self.number_of_zeros_to_add(self.modes[1], self.filter_dim[1])
        padded_input_dim = (_input.shape[2]+ver_pad, _input.shape[3]+hor_pad)
        output_dim = self.get_output_dim(padded_input_dim, self.filter_dim, self.strides)

        # Initialize
        s_output = np.zeros((_input.shape[0], self.num_filters, output_dim[0], output_dim[1]))
        top_pad = 0
        left_pad = 0

        # Iterates through minibatch, filters and incoming channels
        for mb in range(_input.shape[0]):
            # Add padding for minibatch case
            padded_input, top_pad, left_pad = self.add_padding2D(_input[mb], self.modes, self.filter_dim)
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    # Passes a 2D image and a 2D filter to convolution method
                    temp_output = self.conv_parser2D(padded_input[ch], self.filters[f, ch], self.strides)
                    # The convoluted results are summed over each output channel
                    s_output[mb, f] += temp_output

        # Run through activation function
        _output = self.act(s_output)

        # Cache for backward pass and verbose
        self.s_output = s_output
        self.output = _output
        self.left_pad = left_pad
        self.top_pad = top_pad
        return _output

    def backward_pass(self, J_L_Z):
        # Make sure the incoming loss gradient has dimensions (minibatch x num_filters x out_dim0 x out_dim1)
        if np.ndim(J_L_Z) == 2 or np.ndim(J_L_Z) == 3:
            J_L_Z = np.reshape(J_L_Z, (J_L_Z.shape[0], self.num_filters, self.output.shape[2], self.output.shape[3]))

        J_Y_sum = self.act(self.s_output, get_derivative=True)
        J_L_Y_sum = J_L_Z * J_Y_sum


        image_dim = self.input.shape[2:4]

        dL_W = np.zeros((J_L_Z.shape[0], self.num_filters, self.input_channels, self.filter_dim[0], self.filter_dim[1]))
        dL_X = np.zeros((J_L_Z.shape[0], self.input_channels, image_dim[0], image_dim[1]))

        out_dim = J_L_Y_sum.shape[2:4]

        # Iterates through minibatch, filters, incoming channels and image dimensions.
        for mb in range(self.input.shape[0]):
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    k_r = 0
                    for r_loc in range(-self.top_pad, image_dim[0], self.strides[0]):
                        k_c = 0
                        for c_loc in range(-self.left_pad, image_dim[1], self.strides[1]):
                            for r in range(0, self.filter_dim[0], 1):
                                r_in = r_loc + r
                                for c in range(0, self.filter_dim[1], 1):
                                    c_in = c_loc + c
                                    if 0 <= r_in < image_dim[0] and 0 <= c_in < image_dim[1]:
                                        if k_r < out_dim[0] and k_c < out_dim[1]:
                                            dL_W[mb, f, ch, r, c] += self.input[mb, ch, r_in, c_in] * J_L_Y_sum[mb, f, k_r, k_c]
                                            dL_X[mb, ch, r_in, c_in] += self.filters[f, ch, r, c] * J_L_Y_sum[mb, f, k_r, k_c]
                            k_c += 1
                        k_r += 1
        self.dL_W = dL_W.mean(axis=0)
        return dL_X

    def update_weights(self, lr):
        self.filters -= lr * self.dL_W

    def conv_parser2D(self, _input, kernel, stride):
        res_dim = (math.floor((_input.shape[0] - kernel.shape[0]) / stride[0]) + 1,
                   math.floor((_input.shape[1] - kernel.shape[1]) / stride[1]) + 1)
        res = np.zeros(res_dim)
        res_i = 0
        for i in range(0, _input.shape[0], stride[0]):
            res_j = 0
            for j in range(0, _input.shape[1], stride[1]):
                excerpt = _input[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                if excerpt.shape == kernel.shape:
                    value = np.sum(excerpt * kernel)
                    res[res_i, res_j] = value
                res_j += 1
            res_i += 1
        return res

    def add_padding2D(self, _input, modes, kernel_dim):
        ver_pad = self.number_of_zeros_to_add(modes[0], kernel_dim[0])
        hor_pad = self.number_of_zeros_to_add(modes[1], kernel_dim[1])
        top_pad = math.floor(ver_pad / 2)
        left_pad = math.floor(hor_pad / 2)

        channels = _input.shape[0]
        ver_input_len = _input.shape[1]
        hor_input_len = _input.shape[2]

        padded_input = np.zeros((channels, ver_input_len + ver_pad, hor_input_len + hor_pad))
        padded_input[0:channels, top_pad: ver_input_len + top_pad, left_pad: hor_input_len + left_pad] = _input
        return padded_input, top_pad, left_pad


    def get_output_dim(self, padded_input_dim, k_dim, stride):
        res_dim = (math.floor((padded_input_dim[0] - k_dim[0]) / stride[0]) + 1,
                   math.floor((padded_input_dim[1] - k_dim[1]) / stride[1]) + 1)
        return res_dim

    def show_hinton(self):
        if self.do_show_hinton:
            for f in range(self.num_filters):
                for ch in range(self.input_channels):
                    m = self.filters[f, ch]
                    hinton(m)
                    plt.title(f'Layer {self.layer_index}: 2D Conv   Filter shape: {self.filters.shape}\n'
                              f'Filter {f + 1}/{self.num_filters} in input channel {ch + 1}/{self.input_channels}')
                    plt.show()

    def show_verbose(self):
        if self.verbose:
            print('---------------------------')
            print(f'L{self.layer_index}: 2D CONV ')
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


