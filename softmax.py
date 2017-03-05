import numpy as np

class Softmax(object):
    def __init__(self, feature_dim,  classes=10):
        sigma = 2.0 / ( feature_dim + classes)
        W = np.random.normal(loc=0.0,
                             scale=sigma,
                             size=(feature_dim, classes))
        b = np.random.normal(loc=0.001,
                             scale=0.0001,
                             size=(classes,))
        self.params = [W, b]
        self.feature_dim = feature_dim

        def softmax(x):
            max_val = np.max(x)
            x = x - max_val
            return np.exp(x) / np.sum( np.exp(x), axis=0 )

        def softmax_derivative(vector):
            return np.diag(vector) - np.dot(np.expand_dims(vector, 1),
                                            np.expand_dims(vector, 0))
        self.activation_fn = softmax
        self.activation_derivative = softmax_derivative


    def forward(self, feature_vec):
        assert self.feature_dim == feature_vec.shape[-1] # checking the last dim matches or not
        self.inp_vec = feature_vec

        W, b = self.params
        affine_transform = np.dot(feature_vec, W) + b
        probs = self.activation_fn(affine_transform)
        self.deriv_out = self.activation_derivative(affine_transform)

        return probs

    def backward(self, deltas):
        W = self.params[0]

        gdY = self.deriv_out
        dE_dX = np.dot(np.dot(W, gdY),
                       deltas)
        print dE_dX

        dE_dW = np.dot(np.expand_dims(self.inp_vec,
                                      axis=1),
                       np.expand_dims(np.dot(gdY, deltas),
                                      axis=0))
        print dE_dW

        dE_db = np.dot(gdY, deltas)
        print dE_db

        return dE_dX

if __name__ == '__main__':
    feature_dim = 5
    classes = 2
    s = Softmax(feature_dim=feature_dim, classes=classes)
    sample_input = np.ones((feature_dim,))
    print s.forward(sample_input)
    s.backward(np.ones((classes)))
