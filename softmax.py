import numpy as np
import time


class Softmax(object):
    def __init__(self, feature_dim,  classes=10):
        sigma = 2.0 / (feature_dim + classes)
        W = np.random.normal(loc=0.0,
                             scale=sigma,
                             size=(feature_dim, classes))
        b = np.random.normal(loc=0.001,
                             scale=0.0001,
                             size=(classes,))
        dE_dW = np.ndarray(W.shape)
        dE_db = np.ndarray(b.shape)
        self.params = [W, b]
        self.gradParams = [dE_dW, dE_db]
        self.feature_dim = feature_dim
        self.accumulated_gradParams = []
        self.acc_no = 0
        self.name = 'softmax'
        self.momentum_init = False
        self.i_t = 0

        def softmax(x):
            max_val = np.max(x)
            x = x - max_val
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        def softmax_derivative(probs):
            return np.diag(probs) - np.dot(np.expand_dims(probs, 1),
                                           np.expand_dims(probs, 0))
        self.activation_fn = softmax
        self.activation_derivative = softmax_derivative

    def forward(self, feature_vec):
        # checking the last dim matches or not
        start_time = time.time()
        assert self.feature_dim == feature_vec.shape[-1]
        self.inp_vec = feature_vec

        W, b = self.params
        affine_transform = np.dot(feature_vec, W) + b
        probs = self.activation_fn(affine_transform)
        self.deriv_out = self.activation_derivative(probs)
        # print np.sum(W), np.sum(b)
        self.time_taken = time.time() - start_time
        return probs

    def accumulate_grads(self):
        self.acc_no += 1
        if self.accumulated_gradParams == []:
            self.accumulated_gradParams = map(lambda x: x * 1,
                                              self.gradParams)
        else:
            self.accumulated_gradParams = map(lambda x, y: x+y,
                                              self.accumulated_gradParams,
                                              self.gradParams)

    def backward(self, deltas):
        W = self.params[0]
        dE_dW, dE_db = self.gradParams

        gdY = self.deriv_out
        # print np.sum(gdY), np.sum(deltas)
        # print np.sum(dE_dW), 's+++', np.sum(dE_db)
        dE_dX = np.dot(np.dot(W, gdY), deltas)

        dE_dW = np.dot(np.expand_dims(self.inp_vec, axis=1),
                       np.expand_dims(np.dot(gdY, deltas), axis=0))

        dE_db = np.dot(gdY, deltas)
        # print np.sum(dE_dW), 's---', np.sum(dE_db)

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
        # print np.sum(v_t[0]), np.sum(v_t[1])

        self.params = map(lambda theta, m, v:
                          np.subtract(theta,
                                      alpha*np.divide(m,
                                                      np.sqrt(v + epsilon))),
                          self.params, m_t, v_t)

        self.accumulated_gradParams = []


if __name__ == '__main__':
    feature_dim = 5
    classes = 2
    s = Softmax(feature_dim=feature_dim, classes=classes)
    sample_input = np.ones((feature_dim,))
    print s.forward(sample_input)
    s.backward(np.ones((classes)))
    s.backward(np.ones((classes))*10e-5)
    s.updateParams([0.01, 0.9, 0.99])
