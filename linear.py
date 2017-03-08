import numpy as np
import time


class Linear(object):
    def __init__(self,
                 inp_dim, out_dim,
                 activation_fn='relu'):

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.accumulated_gradParams = []
        self.acc_no = 0
        self.name = 'linear'
        self.momentum_init = False
        self.i_t = 0

        sigma = 2.0 / (inp_dim + out_dim)
        W = np.random.normal(loc=0.0,
                             scale=sigma,
                             size=(inp_dim, out_dim))
        b = np.random.normal(loc=0.001,
                             scale=0.0001,
                             size=(out_dim,))
        self.params = [W, b]

        if activation_fn == 'relu':
            self.activation_fn = lambda x: x * (x > 0)
            self.activation_derivative = lambda x: 1. * (x > 0)
        elif activation_fn == 'tanh':
            self.activation_fn = np.tanh
            self.activation_derivative = lambda x: 1.0 - np.tanh(x)**2
        else:
            raise NotImplementedError()

    def forward(self, inp_vec):
        '''This function takes a vector and perform
        an affine transformation followed by non-linearity.
        '''
        start_time = time.time()
        W, b = self.params
        assert inp_vec.shape[-1] == self.inp_dim    # comparing last dimension.
        self.inp_vec = inp_vec

        affine_transformation = np.dot(inp_vec, W) + b
        self.deriv_out = self.activation_derivative(affine_transformation)
        output = self.activation_fn(affine_transformation)

        self.time_taken = time.time() - start_time
        return output

    def accumulate_grads(self):
        self.acc_no += 1
        if self.accumulated_gradParams == []:
            self.accumulated_gradParams = map(lambda x: x * 1,
                                              self.gradParams)
        else:
            self.accumulated_gradParams = map(lambda x, y: x + y,
                                              self.accumulated_gradParams,
                                              self.gradParams)

    def backward(self, deltas):
        '''This function takes as input a vector of
        dimension out_dim, storing dE_dX
        '''
        W = self.params[0]

        gdY = self.deriv_out
        dE_dX = np.dot(np.multiply(W, gdY), deltas)

        dE_dW = np.dot(np.expand_dims(self.inp_vec,
                                      axis=1),
                       np.expand_dims(np.multiply(gdY, deltas),
                                      axis=0))
        dE_db = np.multiply(gdY, deltas)

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
            # print np.sum(self.momentum1[0]), np.sum(self.gradParams[0])

        m_t = map(lambda x: x/(1 - beta1**self.i_t), self.momentum1)
        v_t = map(lambda x: x/(1 - beta2**self.i_t), self.momentum2)

        self.params = map(lambda theta, m, v:
                          np.subtract(theta,
                                      alpha * np.divide(m,
                                                        np.sqrt(v + epsilon))),
                          self.params, m_t, v_t)


if __name__ == '__main__':
    inp_dim = 7
    out_dim = 3
    l = Linear(inp_dim=inp_dim,
               out_dim=out_dim)
    sample_input = np.ones((inp_dim,))
    print l.forward(sample_input).shape
    print l.backward(np.ones(out_dim)).shape
    l.updateParams([0.01, 0.9, 0.99])
    print l.backward(np.ones(out_dim)).shape
    l.updateParams([0.01, 0.9, 0.99])
    print l.backward(np.ones(out_dim)).shape
    l.updateParams([0.01, 0.9, 0.99])
