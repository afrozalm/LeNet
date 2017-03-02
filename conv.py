import numpy as np
from scipy import signal

class Conv(object):
    def __init__(self, n_input_channels, n_output_channels,
                 kr_size,
                 activation_fn='relu'):
        fan_in = n_input_channels * kr_size**2
        fan_out = 1
        sigma = 2.0 / (fan_out+fan_in)
        W = np.random.normal(loc=0.0,
                             scale=sigma,
                             size=( n_output_channels, n_input_channels, kr_size, kr_size))
        b = np.random.normal(loc=0.001,
                             scale=0.0001,
                             size=(n_output_channels))
        self.params = [W, b]
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.kr_size = kr_size
        if activation_fn == 'relu':
            self.activation_fn = lambda x: x * ( x > 0)
            self.activation_derivative = lambda x: 1. * (x > 0)
        elif activation_fn == 'tanh':
            self.activation_fn = np.tanh
            self.activation_derivative = lambda x: 1.0 - np.tanh(x)**2
        else:
            raise NotImplementedError()

    def forward(self, in_fmap):

        '''
        This function takes as input a feature map with dimensions (n_input_channels, [some_width, some_height]). The output is convolution if input feature map with kernels. Convolution from scipy's convolve2d flips kernels in x and y axis then operates.
        '''
        W, b = self.params
        assert in_fmap.shape[0] == self.n_input_channels
        self.in_fmap = in_fmap
        out_shape = (self.n_output_channels,) \
                    + signal.convolve2d(in_fmap[0],
                                        W[0][0],
                                        mode='valid').shape
        out_fmap = np.zeros(out_shape)
        deriv_out = np.ndarray(out_shape)

        for i in xrange(self.n_output_channels):
            for j in xrange(self.n_input_channels):
                out_fmap[i] += b[i] +\
                               signal.convolve2d(in_fmap[j],
                                                 W[i][j],
                                                 mode='valid')

            deriv_out[i] = self.activation_derivative(out_fmap[i])
            out_fmap[i]  = self.activation_fn(out_fmap[i])

        self.deriv_out = deriv_out
        return out_fmap


    def backward(self, deltas):
        '''
        This function takes an input deltas (dE_dX) from succeding layer
        and gives out deltas with respect to current layer.
        '''
        W = self.params[0]
        dE_dX = np.zeros(self.in_fmap.shape)
        dE_dW = np.ndarray(W.shape)
        dE_db = np.ndarray(self.n_output_channels)

        def flip_kr(kernel):
            return np.flipud(np.fliplr(kernel))

        for i in xrange(self.n_output_channels):
            gdY = np.multiply(deltas[i], self.deriv_out[i])
            for j in xrange(self.n_input_channels):
                dE_dX[j] += signal.convolve2d(deltas[i], flip_kr(W[i][j]))
                # Calculating dE_dW
                dE_dW[i][j] = signal.convolve2d(self.in_fmap[j], flip_kr(gdY), mode='valid' )

            # Calculating dE_db
            dE_db[i] = np.sum(gdY)

        self.gradParams = [dE_dW, dE_db]

        return dE_dX


if __name__ == '__main__':
    n_input_channels = 3
    n_output_channels = 4
    c = Conv(n_input_channels, n_output_channels, 2, )
    sample_deltas = np.ndarray((n_output_channels))
    input_map = np.ones((n_input_channels, 32, 32))
    output = c.forward(input_map)
    print output.shape
    sample_deltas = np.ones(output.shape)
    dE_dX = c.backward(sample_deltas)
    print np.sum(dE_dX)
