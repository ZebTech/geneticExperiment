# -*- coding: utf-8 -*-

import random

GOAL = 0.8



class GeneticAlgorithm():
    def __init__(self, pred, ind):
        self.predictor = pred
        self.optimal_individual = pred if pred else Individual()
        self.population = Population(self.op)

    def find_optimal(self):
        generation = 0
        while self.predictor.predict() < GOAL:
            self.population = self.population.evolve()
            self.optimal_individual = self.population.findOptimal(
                self.predictor
            )
            print 'Generation {}, best score: {}' % (
                generation,
                self.predictor.predict(self.optimal_individual)
            )

    @staticmethod
    def get_random_binary():
        return int(random.random())

    @staticmethod
    def get_random_float():
        return random.random()

    @staticmethod
    def get_random_int(limit):
        return int(random.uniform(0, limit))


class Individual():
    pass


class Population():
    pass
