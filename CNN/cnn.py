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
from theano.tensor.signal import downsample

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
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = self.init_weights(filter_shape, poolsize)
        self.b = self.init_bias(filter_shape)
        self.output = self.define_output(filter_shape, image_shape, poolsize)
        self.params = [self.W, self.b]

    def define_output(self, filter_shape, image_shape, poolsize):
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def init_weights(self, filter_shape, poolsize):
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        return theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

    def init_bias(self, filter_shape):
        b_values = np.zeros((filter_shape[0], ), dtype=theano.config.floatX)
        return shared(values=b_values, borrow=True)


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