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


class LogisticRegression(object):
    def __init__(self):
        pass


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
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
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


class CNN(object):
    """
        Implementation of a CNN, inspired from DL Tutorial.
        alpha = learning rate
        epochs = number of traiing epochs
        nkerns = number of kernels on each layer
        batch_size = the size of the training batches
    """
    def __init__(self, alpha=0.1, epochs=200, nkerns=[20, 50], batch_size=500):
        self.fetch_sets()
        rng = np.random.RandomState(1324)
        x = T.matrix('x')
        y = T.ivector('y')
        input0 = x.reshape(batch_size, 1, 28, 28)
        layer0 = ConvPoolLayer(
            rng=rng,
            input=input0,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        layer1 = ConvPoolLayer(
            rng=rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        input2 = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            rng=rng,
            input=input2,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )
        layer3 = LogisticRegression(
            input=layer2.output,
            n_in=500,
            n_out=10
        )

    def fetch_sets(self):
        mnist = fetch_mldata('MNIST original')
        self.trainX = mnist.data[0:TRAINING_SIZE]
        self.trainY = mnist.target[0:TRAINING_SIZE]
        self.testX = mnist.data[TRAINING_SIZE:TRAINING_SIZE + 2000]
        self.testY = mnist.target[TRAINING_SIZE:TRAINING_SIZE + 2000]

    def train(self):
        return self.clf.fit(self.trainX, self.trainY)

    def score(self):
        return self.clf.score(self.testX, self.testY)

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