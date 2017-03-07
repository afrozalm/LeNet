import numpy as np
import time
from conv import Conv
from linear import Linear
from softmax import Softmax
from tqdm import tqdm
from pool import MaxPool_2

class LeNet(object):
    def __init__(self, activation='relu'):
        self.activation = activation
        self.no_of_classes = 10

        self.build_model()

    def build_model(self):
        self.conv_layers = []
        self.linear_layers = []
        self.layers = []

        self.conv_layers += [Conv(1, 6, 5, self.activation)]       # 1x28x28 -> 6x24x24
        self.conv_layers += [MaxPool_2()]                          # 6x24x24 -> 6x12x12
        self.conv_layers += [Conv(6, 16, 5, self.activation)]      # 6x12x12 -> 16x8x8
        self.conv_layers += [MaxPool_2()]                          # 16x8x8  -> 16x4x4
        self.conv_layers += [Conv(16, 120, 4, self.activation)]    # 16x4x4  -> 120x1x1

        self.linear_layers += [Linear(120, 84, self.activation)]   # 120 -> 84
        self.linear_layers += [Softmax(84, self.no_of_classes)]    # 84  -> 10

        for l in self.conv_layers + self.linear_layers:
            self.layers += [l]

    def forward(self, image_tensor):
        assert image_tensor.shape == (1, 28, 28)
        self.inp_tensor = image_tensor
        out = image_tensor
        for l in self.conv_layers:
            out = l.forward(out)

        out = out.reshape(120)
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
        if not target == None:
            self.target = np.zeros(self.no_of_classes)
            self.target[target] = 1

        dE_dX = self.probs - self.target    # derivative of softmax layer w.r.t. cost
        for l in self.linear_layers[::-1]:
            dE_dX = l.backward(dE_dX)

        dE_dX = dE_dX.reshape(120, 1, 1)
        for l in self.conv_layers[::-1]:
            dE_dX = l.backward(dE_dX)

        return dE_dX

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
            if not l.name == 'maxpool':
                l.updateParams(hyperParams)

    def plotPerformance(self, filename):
        trn_pts = []
        val_pts = []

    def test(self, test_data, test_labels):
        hits = 0
        testsize = test_labels.shape[0]
        # testsize = 100
        for inp, target in zip(test_data, test_labels):
            self.forward(inp.reshape(1, 28, 28))
            self.cross_entropy_loss(target)
            hits += self.hit
        return (100.0*hits)/testsize

    def train(self, train_data, train_labels, val_data, val_labels, config):
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
        train_size = train_labels.shape[0]
        val_size = val_labels.shape[0]
        # train_size = 1600
        # val_size = 800
        shuffle_train = np.random.permutation(np.arange(0, train_size))
        shuffle_val   = np.random.permutation(np.arange(0, val_size))

        filename       = config['filename']
        hyperParams    = config['hyperParams']
        minibatch_size = config['batchSize']
        max_epochs     = config['max_epochs']
        assert minibatch_size > 0

        start_time = time.time()
        loss_train = 0
        def TrainStep():
            loss_train = 0
            hits = 0
            for i in tqdm(xrange(0, train_size, minibatch_size )):
                local_loss = 0
                for j in xrange(0, minibatch_size):
                    index = shuffle_train[i + j]
                    self.forward( train_data[index].reshape(1, 28, 28) )
                    local_loss += self.cross_entropy_loss( train_labels[index] )
                    hits += self.hit
                    self.backward()

                if i % 50 == 0 and i > 0:
                    with open('tr_' + filename, 'a') as f:
                        f.write( str(local_loss/minibatch_size) + ', ' \
                                 + str(hits*100.0/i) + ', ' \
                                 + str(time.time() - start_time ) + ', 0\n')

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
                if i % 100 == 0 and i > 0:
                    with open('val_' + filename, 'a') as f:
                        f.write( str(loss_val/500) + ', ' \
                                 + str(hits*100.0/500) + ', ' \
                                 + str( time.time() - start_time) + ', 0\n')
            return [loss_val/val_size, (hits*100.0)/val_size]

        for epoch in xrange(0, max_epochs):
            loss_train, accuracy_train = TrainStep()
            with open('tr_' + filename, 'a') as f:
                f.write( str(loss_train) + ', ' \
                         + str(accuracy_train) + ', ' \
                         + str(time.time() - start_time ) + ', 1\n')

            loss_val, accuracy_val = ValidationStep()
            with open('val_' + filename, 'a') as f:
                f.write( str(loss_val) + ', ' \
                         + str(accuracy_val) + ', ' \
                         + str(time.time() - start_time ) + ', 1\n')

        self.plotPerformance(filename)


if __name__ == '__main__':
    inp_tensor = np.zeros((1, 28, 28))
    net = LeNet()
    net.forward(inp_tensor)
    net.cross_entropy_loss(4)
    net.backward(7)
    net.backward(7)
    net.backward(7)
    net.backward(7)
    net.updateParams([1, 0.9, 0.999])
