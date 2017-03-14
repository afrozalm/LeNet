from keras.datasets import mnist
from tsne import bh_sne
from matplotlib.pyplot import scatter, show
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


class LeNet(object):
    def __init__(self, activation='relu'):
        self.activation = activation
        self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(6, 5, 5,
                                border_mode='valid',
                                input_shape=(1, 28, 28),
                                dim_ordering='th',
                                activation=self.activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, 5, 5,
                                activation=self.activation,
                                dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(120, activation=self.activation))
        model.add(Dense(84, activation=self.activation))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model

    def get_activations(self, X_batch, layer_idx=7):
        K.learning_phase
        get_activations = K.function([self.model.layers[0].input,
                                      K.learning_phase()],
                                     [self.model.layers[layer_idx].output, ])
        activations = get_activations([X_batch, 0])
        return activations

    def train(self, X_train, y_train, X_val, y_val,
              epochs=10, batch_size=200, verbose=2):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       nb_epoch=epochs, batch_size=batch_size, verbose=verbose)

    def test(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def get_hidden_tsne(self, imgs, lbls):
        X = np.asarray(self.get_activations(imgs), dtype=np.float64)
        X_2d = bh_sne(X[0])
        print lbls.shape, X_2d.shape
        scatter(X_2d[:, 0], X_2d[:, 1], c=lbls)
        show()


np.random.seed(1729)
trainset, valset, testset = mnist.load_data()
tr_imgs = trainset[0].reshape(trainset[0].shape[0],
                              1, 28, 28).astype('float32')
vl_imgs = valset[0].reshape(valset[0].shape[0], 1, 28, 28).astype('float32')
te_imgs = testset[0].reshape(testset[0].shape[0], 1, 28, 28).astype('float32')

tr_lbls = np_utils.to_categorical(trainset[1])
vl_lbls = np_utils.to_categorical(valset[1])
te_lbls = np_utils.to_categorical(testset[1])

net = LeNet()
net.train(tr_imgs[:10000], tr_lbls[:10000],
          vl_imgs[:800], vl_lbls[:800],
          batch_size=128, epochs=6)
net.get_hidden_tsne(tr_imgs[:10000], trainset[1][:10000])
