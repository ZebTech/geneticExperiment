# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata


class CNN():
    def __init__(self):
        self.clf = LogisticRegression(penalty='l1', C=10000.0)
        mnist = fetch_mldata('MNIST original')
        self.X = mnist.data[2000:8000]
        self.y = mnist.target[2000:8000]

    def train(self):
        return self.clf.fit(self.X, self.y)

    def score(self):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data[8000:12000]
        y = mnist.target[8000:12000]
        return self.clf.score(X, y)

    def predict(self, features):
        return self.clf.predict_proba(features)
