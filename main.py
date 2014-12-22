# -*- coding: utf-8 -*-
from GA.ga import GeneticAlgorithm
from CNN.cnn import CNN


class Predictor():
    def __init__(self):
        self.clf = CNN()
        self.clf.train()
        print 'Naive test score: ' + str(self.clf.score())

    def predict(self, individual):
        number = 1
        score = self.clf.predict(individual.genes)
        return 2 * score[0][number] - 1.0


pred = Predictor()
ga = GeneticAlgorithm(pred)
opt = ga.find_optimal()
print 'Found optimal:' + opt.to_string()
