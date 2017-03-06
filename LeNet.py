import numpy as np
from conv import Conv
from linear import Linear
from softmax import Softmax
from pool import MaxPool_2

class LeNet(object):
    def __init__(self, activation='relu'):
        self.params = []
        self.gradParams = []
        self.activation = activation
        self.no_of_classes = 10
        self.update_idx = 0
        self.momentum_init = False

        self.build_model()

    def build_model(self):
        self.conv_layers = []
        self.linear_layers = []

        self.conv_layers += [Conv(1, 6, 5, self.activation)]
        self.conv_layers += [MaxPool_2()]
        self.conv_layers += [Conv(6, 16, 5, self.activation)]
        self.conv_layers += [MaxPool_2()]
        self.conv_layers += [Conv(16, 120, 5, self.activation)]

        self.linear_layers += [Linear(120, 84, self.activation)]
        self.linear_layers += [Softmax(84, self.no_of_classes)]

        for l in self.conv_layers + self.linear_layers:
            if not l.name == 'maxpool':
                self.params += [l.params]

    def forward(self, image_tensor):
        assert image_tensor.shape == (1, 32, 32)
        self.inp_tensor = image_tensor
        self.outputs = []
        out = image_tensor
        for l in self.conv_layers:
            out = l.forward(out)
            self.outputs += [out]

        out = out.reshape(120)
        for l in self.linear_layers:
            out = l.forward(out)
            self.outputs += [out]

        return out

    def cross_entropy_loss(self, target):
        epsilon = 10e-20
        assert target < self.no_of_classes
        self.target = np.zeros(self.no_of_classes)
        self.target[target] = 1
        probs = self.outputs[-1]
        if np.argmax(probs) == target:
            self.hit = 1
        else:
            self.hit = 0
        self.loss = -1 * np.log(probs[target] + epsilon)
        return self.loss

    def backward(self, target=None):
        self.gradParams = []
        if not target == None:
            self.target = np.zeros(self.no_of_classes)
            self.target[target] = 1

        dE_dX = self.outputs[-1] - self.target    # derivative of softmax layer w.r.t. cost
        for l in self.linear_layers[::-1]:
            dE_dX = l.backward(dE_dX)
            self.gradParams = [l.gradParams] + self.gradParams

        dE_dX = dE_dX.reshape(120, 1, 1)
        for l in self.conv_layers[::-1]:
            dE_dX = l.backward(dE_dX)
            if not l.name == 'maxpool':
                self.gradParams = [l.gradParams] + self.gradParams

        return dE_dX

    def updateParams(self, hyperParams, optimizer='adam'):
        '''
        This function is used for updating parameters.
        Expects the following input
        hyperParams : (list) a list of hyperParams for the type of optimizer
        optimizer   : (str) 'momentum' or 'adam'
            momentum:
             Takes the following params as a list:
             learningRate : hyperParams[0]
             gamma        : hyperParams[1]

            adam:
             Takes the following params as a list:
             alpha : hyperParams[0]
             beta1 : hyperParams[1]
             beta2 : hyperParams[2]
        '''
        def adam(hyperParams):
            alpha = hyperParams[0]
            beta1 = hyperParams[1]
            beta2 = hyperParams[2]
            epsilon = 10e-20

            self.update_idx += 1
            i_t = self.update_idx

            if not self.momentum_init:
                self.momentum_init = True
                self.momentum1 = []
                self.momentum2 = []
                for g in self.gradParams:
                    self.momentum1.append([np.copy(g[0]),
                                           np.copy(g[1])])
                    self.momentum2.append([np.copy(np.square(g[0])),
                                           np.copy(np.square(g[1]))])
            else:
                self.momentum1 = map( lambda x, y: [beta1 * x[0] + (1 - beta1)*y[0],
                                                    beta1 * x[1] + (1 - beta1)*y[1]],
                                      self.momentum1, self.gradParams)
                self.momentum2 = map( lambda x, y: [beta2 * x[0] + (1 - beta2)*(np.square(y[0])),
                                                    beta2 * x[1] + (1 - beta2)*(np.square(y[1]))],
                                      self.momentum2, self.gradParams)

            m_t = map(lambda x: [x[0]/(1 - beta1**i_t), x[1]/(1 - beta1**i_t)],
                       self.momentum1)
            v_t = map(lambda x: [x[0]/(1 - beta2**i_t), x[1]/(1 - beta2**i_t)],
                       self.momentum2)
            self.params = map( lambda theta, m, v:
                               [theta[0] - alpha * np.divide(m[0], np.sqrt(v[0] + epsilon)),
                                theta[1] - alpha * np.divide(m[1], np.sqrt(v[1] + epsilon))],
                               self.params, m_t, v_t)

        def momentum(hyperParams):
            learningRate = hyperParams[0]
            gamma        = hyperParams[1]
            if not self.momentum_init:
                self.momentum_init = True
                self.accumulated_momentum = []
                for gparam in self.gradParams:
                    self.accumulated_momentum.append([np.copy(gparam[0]),
                                                      np.copy(gparam[1])] )
            else:
                self.accumulated_momentum =  map(lambda x, y: [gamma * x[0] + (1-gamma) * y[0],
                                                               gamma * x[1] + (1-gamma) * y[1]],
                                                 self.accumulated_momentum, self.gradParams )

            self.params = map(lambda x, y: [x[0] - learningRate*y[0], x[1] - learningRate*y[1]],
                              self.params, self.accumulated_momentum)

        if optimizer == 'adam':
            adam(hyperParams)
        else:
            momentum(hyperParams)

    # def plotPerformance(self):

    def test(self, test_data, test_labels):
        hits = 0
        testsize = test_labels.shape[0]
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
         optimizer  : 'momentum' or 'adam'
         hyperParams:
          momentum:
             learningRate : hyperParams[0]
             gamma        : hyperParams[1]
          adam:
             beta1 : hyperParams[0]
             beta2 : hyperParams[1]
             alpha : hyperParams[2]
        '''
        train_size = train_labels.shape[0]
        val_size = val_labels.shape[0]
        shuffle_train = np.random.permutation(np.arange(0, train_size))
        shuffle_val   = np.random.permutation(np.arange(0, val_size))

        filename       = config['filename']
        hyperParams    = config['hyperParams']
        optimizer      = config['optimizer']
        minibatch_size = config['batchSize']
        max_epochs     = config['max_epochs']
        assert minibatch_size > 0

        def accumulate_grads( g1, g2 ):
            if g1 == []:
                return g2
            else:
                return map( lambda x, y: [x[0] + y[0],
                                        x[1] + y[1]],
                            g1, g2)
        loss_train = 0
        def TrainStep():
            loss_train = 0
            hits = 0
            for i in range(0, train_size, minibatch_size ):
                local_gradParams = []
                local_loss = 0
                for j in range(0, minibatch_size):
                    index = shuffle_train[i + j]
                    self.forward( train_data[index])
                    local_loss += self.cross_entropy_loss( train_labels[index] )
                    hits += self.hit
                    self.backward()
                    local_gradParams =  accumulate_grads(local_gradParams, self.gradParams)
                if i % 500 == 0:
                    with open('tr_' + filename, 'a') as f:
                        f.write( str(local_loss/minibatch_size) + ', ' \
                                 + str(hits*100.0/minibatch_size) + '\n')

                # averaging gradParams
                self.gradParams = map(lambda x:[x[0]/minibatch_size,
                                                 x[1]/minibatch_size],
                                      local_gradParams)
                self.updateParams(hyperParams, optimizer)
                loss_train += local_loss
            return [loss_train/train_size, (hits*100.0)/train_size]

        def ValidationStep():
            loss_val = 0
            hits = 0
            for i in range(0, val_size):
                self.forward(val_data[shuffle_val[i]])
                loss_val += self.cross_entropy_loss(val_labels[shuffle_val[i]])
                hits += self.hit
                if i % 500 == 0:
                    with open('val_' + filename, 'a') as f:
                        f.write( str(loss_val/i) + ', ' \
                                 + str(hits*100.0/i) + '\n')
            return [loss_val/val_size, (hits*100.0)/val_size]

        for epoch in tqdm(xrange(0, max_epochs)):
            loss_train, accuracy_train = TrainStep()
            loss_val, accuracy_val = ValidationStep()


if __name__ == '__main__':
    inp_tensor = np.zeros((1, 32, 32))
    net = LeNet()
    net.forward(inp_tensor)
    net.cross_entropy_loss(4)
    net.backward(7).shape
    net.updateParams([0.01, 0.9, 0.999])
