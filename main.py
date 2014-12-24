# -*- coding: utf-8 -*-
from GA.ga import GeneticAlgorithm
from CNN.cnn import CNN
from utils.utils import save_fig
from threading import Lock

NB_IMAGES = 1000
RUN_INSTANCE = 666 # Used to check the existance of a previous learning instance

class Predictor():
    def __init__(self, cnn, number):
        self.clf = cnn
        self.number = number
        self.lock = Lock()
        print 'Naive test score: ' + str(1 - self.clf.score())

    def predict(self, individual):
        if not individual or not individual.genes or None in individual.genes:
            return -1.0
        self.lock.acquire()
        score = self.clf.predict([individual.genes, ])
        self.lock.release()
        return 2 * score[0][self.number] - 1.0


cnn = CNN(epochs=100, batch_size=100, instance_id=RUN_INSTANCE)
cnn.train()
for i in xrange(NB_IMAGES):
    for i in xrange(10):
        pred = Predictor(cnn, i)
        ga = GeneticAlgorithm(pred)
        opt = ga.find_optimal()
        print 'Found optimal:' + opt.to_string()
        save_fig(opt.genes, str(i))
