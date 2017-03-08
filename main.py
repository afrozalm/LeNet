import gzip
import cPickle as pkl
from LeNet import LeNet

f = gzip.open('mnist.pkl.gz', 'rb')
trainset, valset, testset = pkl.load(f)
f.close()

net = LeNet()
config = {
    'filename': 'relu.log',
    'batchSize': 80,
    'max_epochs': 4,
    'hyperParams': [0.0001, 0.9, 0.999]
}

net.train(trainset[0], trainset[1], valset[0], valset[1], config)
# print '['+config['filename']+'] Accuracy on test set is ' \
#     + str( net.test(testset[0], testset[1])) + '%'
