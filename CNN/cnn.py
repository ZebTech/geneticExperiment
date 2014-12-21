# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

class CNN():
    def __init__(self):
        self.clf = LogisticRegression()
        digits = load_digits()
        self.X = (digits.data/16).astype(int)
        self.y = digits.target

    def train(self):
        return self.clf.fit(self.X, self.y)

    def score(self):
        return self.clf.score(self.X, self.y)

    def predict(self, features):
        return self.clf.predict_proba(features)
