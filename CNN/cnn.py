# -*- coding: utf-8 -*-
import warnings
import time
import theano
import pickle as pickle
import theano.tensor as T
import numpy as np

from sklearn.datasets import fetch_mldata
from theano import (
    shared,
    function,
)
from theano.tensor.nnet import (
    conv,
    sigmoid,
    softmax
)
from theano.tensor.signal import downsample

warnings.filterwarnings("ignore")

TRAINING_SIZE = 69000
TESTING_SIZE = 1000


class Layer(object):
    def __getstate__(self):
        return (self.W, self.b)

    def __setstate__(self, state):
        W, b = state
        self.W = W
        self.b = b


class LogisticRegression(Layer):
    def __init__(self, input, n_in, n_out):
        self.W = self.init_weights(n_in, n_out)
        self.b = self.init_bias(n_out)
        self.p_y_given_x = softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def init_weights(self, n_in, n_out):
        return shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

    def init_bias(self, n_out):
        return shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )


class HiddenLayer(Layer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        self.W = W if W else self.init_weights(rng, n_in, n_out, activation)
        self.b = b if b else self.init_bias(n_out)
        self.params = [self.W, self.b]

        linear_output = T.dot(input, self.W) + self.b
        self.output = (
            linear_output if activation is None else activation(linear_output)
        )

    def init_weights(self, rng, n_in, n_out, activation):
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


class ConvPoolLayer(Layer):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = self.init_weights(rng, filter_shape, poolsize)
        self.b = self.init_bias(filter_shape)
        self.output = self.define_output(filter_shape, image_shape, poolsize)
        self.params = [self.W, self.b]

    def define_output(self, filter_shape, image_shape, poolsize):
        conv_out = conv.conv2d(
            input=self.input,
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

    def init_weights(self, rng, filter_shape, poolsize):
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
        return shared(value=b_values, borrow=True)


class CNN(object):
    """
        Implementation of a CNN, inspired from DL Tutorial.
        alpha = learning rate
        epochs = number of training epochs
        nkerns = number of kernels on each layer
        batch_size = the size of the training batches
    """
    def __init__(self, alpha=0.1, epochs=200, nkerns=[20, 50], batch_size=500, 
                 instance_id=None):
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.fetch_sets()
        self.init_network(nkerns, batch_size)
        self.instance = instance_id

    def init_network(self, nkerns, batch_size):
        rng = np.random.RandomState(1324)
        x = T.matrix('x')
        y = T.ivector('y')
        input0 = x.reshape((batch_size, 1, 28, 28))
        self.layer0 = ConvPoolLayer(
            rng=rng,
            input=input0,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        self.layer1 = ConvPoolLayer(
            rng=rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        input2 = self.layer1.output.flatten(2)
        self.layer2 = HiddenLayer(
            rng=rng,
            input=input2,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )
        self.layer3 = LogisticRegression(
            input=self.layer2.output,
            n_in=500,
            n_out=10
        )
        cost = self.layer3.negative_log_likelihood(y)
        params = (
            self.layer3.params +
            self.layer2.params +
            self.layer1.params +
            self.layer0.params
        )
        grads = T.grad(cost, params)
        updates = [
            (param_i, param_i - self.alpha * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        index = T.lscalar()
        self.train_model = function(
            [index],
            cost,
            updates=updates,
            givens={
                x: self.trainX[index * batch_size: (index + 1) * batch_size],
                y: self.trainY[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='ignore'
        )

        self.validate_model = function(
            [index],
            self.layer3.errors(y),
            givens={
                x: self.testX[index * batch_size: (index + 1) * batch_size],
                y: self.testY[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='ignore'
        )
        predX = T.matrix('predX')
        self.predict_model = function(
            [predX],
            self.layer3.p_y_given_x,
            givens={
                x: predX[0: batch_size]
            },
            on_unused_input='ignore'
        )

    def fetch_sets(self):
        mnist = fetch_mldata('MNIST original')
        self.trainX = mnist.data[0:TRAINING_SIZE]
        self.trainY = mnist.target[0:TRAINING_SIZE]
        self.testX = mnist.data[TRAINING_SIZE:TRAINING_SIZE + TESTING_SIZE]
        self.testY = mnist.target[TRAINING_SIZE:TRAINING_SIZE + TESTING_SIZE]
        self.trainX = shared(
            np.asarray(self.trainX, dtype=theano.config.floatX),
            borrow=True
        )
        self.trainY = T.cast(
            shared(
                np.asarray(self.trainY, dtype=theano.config.floatX),
                borrow=True),
            'int32'
        )

        self.testX = shared(
            np.asarray(self.testX, dtype=theano.config.floatX),
            borrow=True
        )
        self.testY = T.cast(
            shared(
                np.asarray(self.testY, dtype=theano.config.floatX),
                borrow=True),
            'int32'
        )

    def train(self):
        old_training = 0
        if self.instance:
            old_training = self.load_training_values()
        n_train_batches = TRAINING_SIZE / self.batch_size
        for epoch in xrange(self.epochs - old_training):
            for minibatch_index in xrange(n_train_batches):
                iter = epoch * n_train_batches + minibatch_index
                cost_ij = self.train_model(minibatch_index)
                if (iter + 1) % n_train_batches == 0:
                    validation_loss = self.score()
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           validation_loss * 100.))
                    self.save_network(epoch + old_training)
                    
    def load_training_values(self):
        saved = pickle.load(
            open('layers_' + str(self.instance) + '.pkl', 'wb')
        )
        if saved and len(saved) == 5:
            self.layer0 = saved[4]
            self.layer1 = saved[3]
            self.layer2 = saved[2]
            self.layer3 = saved[1]
            return saved[0]
        return 0
                    
    def save_network(self, epoch):
        pickle.dump([
            epoch,
            self.layer3,
            self.layer2,
            self.layer1,
            self.layer0,
        ], open('layers_' + str(self.instance) + '.pkl', 'wb'))

    def score(self):
        n_test_batches = TESTING_SIZE / self.batch_size
        validation_losses = [self.validate_model(i) for i
                             in xrange(n_test_batches)]
        return np.mean(validation_losses)

    def predict(self, features):
        formated = np.ones(
            (self.batch_size, len(features[0])),
            dtype=theano.config.floatX
        )
        formated[0:len(features)] = features
        result = self.predict_model(formated)
        return result[0:len(features)]


if __name__ == '__main__':
    print 'creating'
    cnn = CNN(epochs=10, batch_size=1000)
    print 'created'
    cnn.train()
    print 'trained'
    m = cnn.testX[25:25+cnn.batch_size]
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    print cnn.predict(X[0:5])
