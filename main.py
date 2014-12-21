# -*- coding: utf-8 -*-
from GA.ga import GeneticAlgorithm


class Predictor():
    def __init__(self):
        self.optimal = [0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]

    def predict(self, individual):
        count = 0
        for i in xrange(len(self.optimal)):
            a = self.optimal[i]
            b = individual.get_gene(i)
            count += 1 if a != b else 0
        return float(count)/float(len(self.optimal))

print 'Started'
pred = Predictor()
ga = GeneticAlgorithm(pred)
print 'Working'
opt = ga.find_optimal()
print 'Found optimal:' + opt.to_string()
