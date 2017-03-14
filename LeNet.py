import numpy as np
from tsne import bh_sne
from matplotlib.pyplot import scatter, show
import time
import matplotlib.pyplot as plt
from conv import Conv
from linear import Linear
from softmax import Softmax
import os
from tqdm import tqdm
from pool import MaxPool_2
from itertools import product


class LeNet(object):
    def __init__(self, activation='tanh'):
        self.activation = activation
        self.no_of_classes = 10
        self.build_model()

    def build_model(self):
        self.conv_layers = []
        self.linear_layers = []
        self.layers = []

        # 1x28x28 -> 6x24x24
        self.conv_layers += [Conv(1, 6, 5, self.activation)]
        # 6x24x24 -> 6x12x12
        self.conv_layers += [MaxPool_2()]
        # 6x12x12 -> 16x8x8
        self.conv_layers += [Conv(6, 16, 5, self.activation)]
        # 16x8x8  -> 16x4x4
        self.conv_layers += [MaxPool_2()]

        # 256 -> 120
        self.linear_layers += [Linear(16*4*4, 120, self.activation)]
        # 120 -> 84
        self.linear_layers += [Linear(120, 84, self.activation)]
        # 84  -> 10
        self.linear_layers += [Softmax(84, self.no_of_classes)]

        self.layers = self.conv_layers + self.linear_layers

    def forward(self, image_tensor):
        assert image_tensor.shape == (1, 28, 28)
        self.inp_tensor = image_tensor
        out = image_tensor
        for l in self.conv_layers:
            out = l.forward(out)

        out = out.reshape(16*4*4)
        for l in self.linear_layers:
            out = l.forward(out)

        self.probs = out
        return out

    def cross_entropy_loss(self, target):
        epsilon = 10e-20
        assert target < self.no_of_classes
        self.target = np.zeros(self.no_of_classes)
        self.target[target] = 1
        if np.argmax(self.probs) == target:
            self.hit = 1
        else:
            self.hit = 0
        self.loss = -1 * np.log(self.probs[target] + epsilon)
        return self.loss

    def backward(self, target=None):
        if target is not None:
            self.target = np.zeros(self.no_of_classes)
            self.target[target] = 1

        # derivative of softmax layer w.r.t. cost
        dE_dX = self.probs - self.target
        for l in self.linear_layers[::-1]:
            dE_dX = l.backward(dE_dX)

        dE_dX = dE_dX.reshape(16, 4, 4)
        for l in self.conv_layers[::-1]:
            dE_dX = l.backward(dE_dX)

        return dE_dX

    def tell_time_taken(self):
        inp_tensor = np.ndarray((1, 28, 28))
        self.forward(inp_tensor)
        c = 1
        l = 1
        for conv in self.conv_layers:
            print 'Time taken for conv' + str(c) + ' = ' + str(conv.time_taken)
            c += 1

        for lin in self.linear_layers:
            print 'Time taken for fc' + str(l) + ' = ' + str(lin.time_taken)
            l += 1

    def get_numeric_grads(self, target=4):
        inp_tensor = np.random.normal(loc=2.0,
                                      scale=0.1,
                                      size=(1, 28, 28))
        self.forward(inp_tensor)
        self.backward(target)
        epsilon = 1e-10

        def get_conv_grad():
            c1 = self.conv_layers[0]
            W, b = self.conv_layers[0].params
            in_chs, out_chs, kr_w, kr_h = W.shape
            calculate_W_grad = np.ndarray(W.shape)
            squaredW = np.ndarray(W.shape)
            for in_ch in xrange(in_chs):
                for out_ch in xrange(out_chs):
                    for i, j in product(xrange(kr_w), repeat=2):
                        orig_value = W[in_ch][out_ch][i][j]
                        self.conv_layers[0].params[0][in_ch][out_ch][i][j] \
                            += epsilon
                        self.forward(inp_tensor)
                        E_plus = self.cross_entropy_loss(target)
                        self.conv_layers[0].params[0][in_ch][out_ch][i][j] \
                            -= 2*epsilon
                        self.forward(inp_tensor)
                        E_minus = self.cross_entropy_loss(target)
                        grad_val = (E_plus - E_minus)/(epsilon*2)
                        calculate_W_grad[in_ch][out_ch][i][j] = grad_val
                        squaredW[in_ch][out_ch][i][j] \
                            = (grad_val - c1.gradParams[0][in_ch][out_ch][i][j])**2
                        self.conv_layers[0].params[0][in_ch][out_ch][i][j] \
                            = orig_value

                        print grad_val
                        print orig_value
                        if not orig_value == 0.0:
                            print abs(grad_val - orig_value)/abs(orig_value)
                        print '--------'

        def get_lin_grad(layer=0):
            l1 = self.linear_layers[layer]
            W, b = self.linear_layers[layer].params
            inp_dim, out_dim = W.shape
            calculate_W_grad = np.ndarray(W.shape)
            squaredW = np.ndarray(W.shape)
            for i in xrange(inp_dim):
                for j in xrange(out_dim):
                    orig_value = W[i][j]
                    self.linear_layers[layer].params[0][i][j] \
                        += epsilon
                    self.forward(inp_tensor)
                    E_plus = self.cross_entropy_loss(target)
                    self.linear_layers[layer].params[0][i][j] \
                        -= 2*epsilon
                    self.forward(inp_tensor)
                    E_minus = self.cross_entropy_loss(target)
                    grad_val = (E_plus - E_minus) / (2*epsilon)
                    calculate_W_grad[i][j] = grad_val
                    squaredW[i][j] \
                        = (grad_val - l1.gradParams[0][i][j])**2
                    self.linear_layers[layer].params[0][i][j] = orig_value

                    print grad_val
                    print orig_value
                    if not orig_value == 0.0:
                        print abs(grad_val - orig_value)/abs(orig_value)
                    print '--------'

        # get_conv_grad()
        get_lin_grad(layer=2)

    def updateParams(self, hyperParams):
        '''
        This function is used for updating parameters.
        Expects the following input
        hyperParams : (list) a list of hyperParams for adam
            adam:
             Takes the following params as a list:
             alpha : hyperParams[0]
             beta1 : hyperParams[1]
             beta2 : hyperParams[2]
        '''
        for l in self.layers:
            if l.name not in ['maxpool']:
                l.updateParams(hyperParams)

    def plotPerformance(self, filename):
        def get_pts(filename):
            with open(filename, 'r') as f:
                data = f.read()
            all_pts = data.split('\n')
            all_pts = all_pts[:-1]  # ignoring last \n
            all_pts = [a.split(',') for a in all_pts]
            all_pts = [[float(b) for b in a] for a in all_pts]

            all_time = []
            all_acc = []
            all_loss = []
            for loss, acc, t in all_pts:
                all_loss += [loss]
                all_acc += [acc]
                all_time += [t]

            return all_loss, all_acc, all_time
        tr_loss, tr_acc, tr_time = get_pts('tr_' + filename)
        val_loss, val_acc, val_time = get_pts('val_' + filename)

        def smoothen(lst, alpha=0.05):
            smooth_lst = []
            prev = lst[0]
            for ele in lst:
                curr = alpha*ele + (1-alpha)*prev
                smooth_lst += [curr]
                prev = curr
            return smooth_lst

        plt.plot(tr_time, smoothen(tr_loss, 0.05))
        plt.plot(val_time, smoothen(val_loss, 0.05))
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.legend(['train loss', 'validation loss'], loc=1)
        plt.show()

        plt.plot(tr_time, smoothen(tr_acc, 1))
        plt.plot(val_time, smoothen(val_acc, 1))
        plt.xlabel('time')
        plt.ylabel('accuracy')
        plt.legend(['train accuracy', 'validation accuracy'], loc=3)
        plt.show()

    def plot_tsne(self, X, y):
        X = np.asarray(self.get_activations(X), dtype=np.float64)
        X_2d = bh_sne(X[0])
        scatter(X_2d[:, 0], X_2d[:, 1], c=y)
        show()

    def get_activations(self, X):
        output = np.ndarray(X.shape[0], 84)
        for i in xrange(X.shape[0]):
            output[i] = self.get_feature_vec(X[i])
        return output

    def get_feature_vec(self, image_tensor):
        assert image_tensor.shape == (1, 28, 28)
        out = image_tensor
        for l in self.conv_layers:
            out = l.forward(out)

        out = out.reshape(16*4*4)
        for l in self.layers:
            if l.name is not 'softmax':
                out = l.forward(out)
            return out

    def test(self, test_data, test_labels):
        hits = 0
        # testsize = test_labels.shape[0]
        testsize = 100
        for inp, target in zip(test_data, test_labels):
            self.forward(inp.reshape(1, 28, 28))
            self.cross_entropy_loss(target)
            hits += self.hit
        return (100.0*hits)/testsize

    def train(self, train_data, train_labels, val_data, val_labels, config):  #
        '''
        config : a dictionary with following entries
         filename   : name of file to store results
         batchSize  : batchSize for minibatch
         max_epochs : number of epochs to run
         hyperParams: a list with -
          momentum:
             learningRate : hyperParams[0]
             gamma        : hyperParams[1]
          adam:
             alpha : hyperParams[0]
             beta1 : hyperParams[1]
             beta2 : hyperParams[2]
        '''
        # train_size = train_labels.shape[0]
        # val_size = val_labels.shape[0]
        train_size = 1600
        val_size = 800
        shuffle_train = np.random.permutation(np.arange(0, train_size))
        shuffle_val = np.random.permutation(np.arange(0, val_size))

        filename = config['filename']
        hyperParams = config['hyperParams']
        minibatch_size = config['batchSize']
        max_epochs = config['max_epochs']
        assert minibatch_size > 0

        def silent_remove(filename):
            try:
                os.remove(filename)
            except OSError:
                pass

        silent_remove('tr_' + filename)
        silent_remove('val_' + filename)

        start_time = time.time()
        loss_train = 0

        def TrainStep():
            loss_train = 0
            hits = 0
            for i in tqdm(xrange(0, train_size, minibatch_size)):
                mini_hits = 0
                local_loss = 0
                for j in xrange(0, minibatch_size):
                    index = shuffle_train[i + j]
                    self.forward(train_data[index].reshape(1, 28, 28))
                    local_loss += self.cross_entropy_loss(train_labels[index])
                    hits += self.hit
                    mini_hits += self.hit
                    self.backward()
                print mini_hits*100.0/minibatch_size
                self.updateParams(hyperParams)
                loss_train += local_loss
            return [loss_train/train_size, (hits*100.0)/train_size]

        def ValidationStep():
            loss_val = 0
            hits = 0
            for i in tqdm(xrange(0, val_size)):
                self.forward(val_data[shuffle_val[i]].reshape(1, 28, 28))
                loss_val += self.cross_entropy_loss(val_labels[shuffle_val[i]])
                hits += self.hit
            return [loss_val/val_size, (hits*100.0)/val_size]

        for epoch in xrange(0, max_epochs):
            # shuffle_train = np.random.permutation(np.arange(0, train_size))
            # shuffle_val = np.random.permutation(np.arange(0, val_size))
            loss_train, accuracy_train = TrainStep()
            with open('tr_' + filename, 'a') as f:
                f.write(str(loss_train) + ', '
                        + str(accuracy_train) + ', '
                        + str(time.time() - start_time) + '\n')

            loss_val, accuracy_val = ValidationStep()
            with open('val_' + filename, 'a') as f:
                f.write(str(loss_val) + ', '
                        + str(accuracy_val) + ', '
                        + str(time.time() - start_time) + '\n')


if __name__ == '__main__':
    inp_tensor = np.random.normal(size=(1, 28, 28))
    net = LeNet()
    # net.tell_time_taken()
    net.plotPerformance('relu.log')
    # net.forward(inp_tensor)
    # net.cross_entropy_loss(4)
    # net.backward(7)
    # net.backward(1)
    # net.backward(4)
    # net.backward(9)
    # net.updateParams([1, 0.9, 0.999])
    # net.get_numeric_grads()
