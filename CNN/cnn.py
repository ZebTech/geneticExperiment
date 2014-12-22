# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata

TRAINING_SIZE = 10000


class CNN():
    def __init__(self):
        self.clf = LogisticRegression(penalty='l1', C=10000.0)
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
