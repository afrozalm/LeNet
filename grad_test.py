import numpy as np
import time
import matplotlib.pyplot as plt
from linear import Linear
from softmax import Softmax
import os
from tqdm import tqdm


class LeNet(object):
    def __init__(self, activation='relu'):
        self.activation = activation
        self.no_of_classes = 10
        self.build_model()

    def build_model(self):
        self.linear_layers = []
        self.layers = []

        self.linear_layers += [Linear(28*28, 120, self.activation)]
        # 120 -> 84
        self.linear_layers += [Linear(120, 84, self.activation)]
        # 84  -> 10
        self.linear_layers += [Softmax(84, self.no_of_classes)]

        self.layers = self.linear_layers

    def forward(self, image_tensor):
        assert image_tensor.shape == (28*28,)
        self.inp_tensor = image_tensor
        out = image_tensor
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
            # print np.mean(dE_dX)
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
            l.updateParams(hyperParams)

    def plotPerformance(self, filename):
        def get_pts(filename):
            with open(filename, 'r') as f:
                data = f.read()
            all_pts = data.split('\\n')
            all_pts = all_pts[:-1]  # ignoring last \n
            all_pts = [a.split(',') for a in all_pts]
            all_pts = [[float(b) for b in a] for a in all_pts]

            all_time = []
            all_acc = []
            all_loss = []
            for loss, acc, t, _ in all_pts:
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

        plt.plot(tr_time, smoothen(tr_acc, 0.05))
        plt.plot(val_time, smoothen(val_acc, 0.05))
        plt.xlabel('time')
        plt.ylabel('loss')
        plt.legend(['train loss', 'validation loss'], loc=1)
        plt.show()

    def test(self, test_data, test_labels):
        hits = 0
        # testsize = test_labels.shape[0]
        testsize = 100
        for inp, target in zip(test_data, test_labels):
            self.forward(inp)
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
                local_loss = 0
                for j in xrange(0, minibatch_size):
                    index = shuffle_train[i + j]
                    self.forward(train_data[index])
                    local_loss += self.cross_entropy_loss(train_labels[index])
                    hits += self.hit
                    self.backward()

                if i % 50 == 0 and i > 0:
                    with open('tr_' + filename, 'a') as f:
                        f.write(str(local_loss/minibatch_size) + ', '
                                + str(hits*100.0/i) + ', '
                                + str(time.time() - start_time) + ', 0\n')

                self.updateParams(hyperParams)
                loss_train += local_loss
            return [loss_train/train_size, (hits*100.0)/train_size]

        def ValidationStep():
            loss_val = 0
            hits = 0
            for i in tqdm(xrange(0, val_size)):
                self.forward(val_data[shuffle_val[i]])
                loss_val += self.cross_entropy_loss(val_labels[shuffle_val[i]])
                hits += self.hit
                if i % 100 == 0 and i > 0:
                    with open('val_' + filename, 'a') as f:
                        f.write(str(loss_val/500) + ', '
                                + str(hits*100.0/500) + ', '
                                + str(time.time() - start_time) + ', 0\n')
            return [loss_val/val_size, (hits*100.0)/val_size]

        for epoch in xrange(0, max_epochs):
            loss_train, accuracy_train = TrainStep()
            with open('tr_' + filename, 'a') as f:
                f.write(str(loss_train) + ', '
                        + str(accuracy_train) + ', '
                        + str(time.time() - start_time) + ', 1\n')

            loss_val, accuracy_val = ValidationStep()
            with open('val_' + filename, 'a') as f:
                f.write(str(loss_val) + ', '
                        + str(accuracy_val) + ', '
                        + str(time.time() - start_time) + ', 1\n')

        # self.plotPerformance(filename)


if __name__ == '__main__':
    inp_tensor = np.zeros((1, 28, 28))
    net = LeNet()
    net.forward(inp_tensor)
    net.cross_entropy_loss(4)
    net.backward(7)
    net.backward(1)
    net.backward(4)
    net.backward(9)
    net.updateParams([1, 0.9, 0.999])
