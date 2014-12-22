# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np

from sklearn.datasets import fetch_mldata
from theano import (
    shared,
)
from theano.tensor.nnet import (
    conv,
    sigmoid,
)

TRAINING_SIZE = 6000


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        self.W = W if W else self.init_weights(n_in, n_out, activation)
        self.b = b if b else self.init_bias(n_out)
        self.params = [self.W, self.b]

        linear_output = T.dot(input, self.W) + self.b
        self.output = (
            linear_output if activation is None else activation(linear_output)
        )

    def init_weights(self, n_in, n_out, activation):
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        if activation == sigmoid:
            W_values *= 4
        return shared(value=W_values, name='W', borrow=True)

    def init_bias(self, n_out):
        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        return shared(value=b_values, name='b', borrow=True)


class ConvPoolLayer(object):
    def __init__(self):
        pass


class CNN():
    def __init__(self):
        mnist = fetch_mldata('MNIST original')
        self.X = mnist.data[0:TRAINING_SIZE]
        self.y = mnist.target[0:TRAINING_SIZE]

    def train(self):
        return self.clf.fit(self.X, self.y)

    def score(self):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data[TRAINING_SIZE:TRAINING_SIZE + 2000]
        y = mnist.target[TRAINING_SIZE:TRAINING_SIZE + 2000]
        return self.clf.score(X, y)

    def predict(self, features):
        return self.clf.predict_proba(features)


if __name__ == '__main__':
    rng = np.random.RandomState(23)
    input = T.matrix('x')
    y = T.ivector('y')
    h = HiddenLayer(rng=rng, input=input, n_in=784, n_out=10)
    print 'created'
#    clf = CNN()
#    clf.train()
#    print 'Naive test score: ' + str(clf.score())