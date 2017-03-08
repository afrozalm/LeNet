import numpy as np
from scipy import signal
import time


class Conv(object):
    def __init__(self, n_input_channels, n_output_channels,
                 kr_size,
                 activation_fn='relu'):
        fan_in = n_input_channels * kr_size**2
        fan_out = n_output_channels * kr_size**2 / 4  # 4 is for pool size
        sigma = 2.0 / (fan_out+fan_in)
        W = np.random.normal(loc=0.0,
                             scale=sigma,
                             size=(n_output_channels,
                                   n_input_channels,
                                   kr_size, kr_size))
        b = np.random.normal(loc=0.001,
                             scale=0.0001,
                             size=(n_output_channels))
        dE_dW = np.ndarray(W.shape)
        dE_db = np.ndarray(b.shape)
        self.name = 'conv'
        self.params = [W, b]
        self.gradParams = [dE_dW, dE_db]
        self.accumulated_gradParams = []
        self.acc_no = 0
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.kr_size = kr_size
        self.momentum_init = False
        self.i_t = 0

        if activation_fn == 'relu':
            self.activation_fn = lambda x: x * (x > 0)
            self.activation_derivative = lambda x: 1. * (x > 0)
        elif activation_fn == 'tanh':
            self.activation_fn = np.tanh
            self.activation_derivative = lambda x: 1.0 - np.tanh(x)**2
        else:
            raise NotImplementedError()

    def forward(self, in_fmap):

        '''
        This function takes as input a feature map with
        dimensions (n_input_channels, [some_width, some_height]).
        The output is convolution if input feature map with kernels.
        Convolution from scipy's convolve2d flips kernels in x and y
        axis then operates.
        '''
        start_time = time.time()
        W, b = self.params
        assert in_fmap.shape[0] == self.n_input_channels
        self.in_fmap = in_fmap
        out_shape = signal.convolve2d(in_fmap[0], W[0][0], mode='valid').shape
        out_shape = (self.n_output_channels,) + out_shape
        out_fmap = np.zeros(out_shape)
        deriv_out = np.ndarray(out_shape)

        for i in xrange(self.n_output_channels):
            for j in xrange(self.n_input_channels):
                out_fmap[i] += b[i] +\
                               signal.convolve2d(in_fmap[j],
                                                 W[i][j],
                                                 mode='valid')

            deriv_out[i] = self.activation_derivative(out_fmap[i])
            out_fmap[i] = self.activation_fn(out_fmap[i])

        self.deriv_out = deriv_out
        self.time_taken = time.time() - start_time
        return out_fmap

    def accumulate_grads(self):
        self.acc_no += 1
        if self.accumulated_gradParams == []:
            self.accumulated_gradParams = map(lambda x: np.copy(x),
                                              self.gradParams)
        else:
            self.accumulated_gradParams = map(lambda x, y: np.add(x, y),
                                              self.accumulated_gradParams,
                                              self.gradParams)

    def backward(self, deltas):
        '''
        This function takes an input deltas (dE_dX) from succeding layer
        and gives out deltas with respect to current layer.
        '''
        W = self.params[0]
        dE_dW, dE_db = self.gradParams
        dE_dX = np.zeros(self.in_fmap.shape)

        def flip_kr(kernel):
            return np.flipud(np.fliplr(kernel))

        # print np.sum(dE_dW), 'c+++', np.sum(dE_db)
        for i in xrange(self.n_output_channels):
            gdY = np.multiply(deltas[i], self.deriv_out[i])
            for j in xrange(self.n_input_channels):
                dE_dX[j] += signal.convolve2d(deltas[i], flip_kr(W[i][j]))
                # Calculating dE_dW
                dE_dW[i][j] = flip_kr(signal.convolve2d(self.in_fmap[j],
                                                        flip_kr(gdY),
                                                        mode='valid'))
            # Calculating dE_db
            dE_db[i] = np.sum(gdY)

        # print np.sum(dE_dW), 'c---', np.sum(dE_db)
        self.gradParams = [dE_dW, dE_db]

        self.accumulate_grads()
        return dE_dX

    def updateParams(self, hyperParams):
        alpha, beta1, beta2 = hyperParams
        epsilon = 10e-20
        self.i_t += 1

        self.gradParams = map(lambda x: x / self.acc_no,
                              self.accumulated_gradParams)
        self.acc_no = 0
        self.accumulated_gradParams = []

        if not self.momentum_init:
            self.momentum_init = True
            self.momentum1 = []
            self.momentum2 = []
            for g in self.gradParams:
                self.momentum1.append(np.copy(g))
                self.momentum2.append(np.copy(np.square(g)))

        else:
            self.momentum1 = map(lambda m, g: np.add(beta1 * m,
                                                     (1 - beta1) * g),
                                 self.momentum1, self.gradParams)
            self.momentum2 = map(lambda m, g: np.add(beta2 * m,
                                                     (1 - beta2) *
                                                     np.square(g)),
                                 self.momentum2, self.gradParams)

        m_t = map(lambda x: x/(1 - beta1**self.i_t), self.momentum1)
        v_t = map(lambda x: x/(1 - beta2**self.i_t), self.momentum2)
        print np.mean(self.params[0]), 'wwwwwwww'
        print np.mean(self.gradParams[0]), 'dddddddddddd'
        print np.mean(v_t[0]), 'vvvvvvvvvvvv'
        print '============='

        self.params = map(lambda theta, m, v:
                          np.subtract(theta,
                                      alpha*np.divide(m,
                                                      np.sqrt(v + epsilon))),
                          self.params, m_t, v_t)


if __name__ == '__main__':
    n_input_channels = 1
    n_output_channels = 6
    c = Conv(n_input_channels, n_output_channels, 5)
    sample_deltas = np.ndarray((n_output_channels))
    input_map = np.ones((n_input_channels, 28, 28))
    output = c.forward(input_map)
    # print output.shape
    sample_deltas = np.ones(output.shape)
    dE_dX = c.backward(sample_deltas * 10e-7)
    dE_dX = c.backward(sample_deltas * 10e-3)
    dE_dX = c.backward(sample_deltas * 10e-2)
    dE_dX = c.backward(sample_deltas * 10e-7)
    # print np.sum(dE_dX)
    c.updateParams([0.01, 0.9, 0.99])
