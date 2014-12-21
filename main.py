# -*- coding: utf-8 -*-
from GA.ga import GeneticAlgorithm
from CNN.cnn import CNN
import numpy as np


class Predictor():
    def __init__(self):
        self.clf = CNN()
        self.clf.train()
        print 'Naive test score: ' + str(self.clf.score())

    def predict(self, individual):
        number = 5
        score = self.clf.predict(individual.genes)
        total = np.sum(score)
        return 2 * score[0][number] - total

print 'Started'
pred = Predictor()
ga = GeneticAlgorithm(pred)
print 'Working'
opt = ga.find_optimal()
print 'Found optimal:' + opt.to_string()
