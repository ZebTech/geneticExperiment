# -*- coding: utf-8 -*-

import random
from random import shuffle
import copy
import ThreadPool

GOAL = 0.8
INDIVIDUAL_SIZE = 10


class GeneticAlgorithm():
    def __init__(self, pred, ind):
        self.predictor = pred
        self.optimal_individual = pred if pred else Individual()
        self.population = Population(self.op)

    def find_optimal(self):
        generation = 0
        while self.predictor.predict() < GOAL:
            self.population = self.population.evolve()
            self.optimal_individual = self.population.find_optimal(
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
    Size = 100
    TruncateProportionalSelection = True
    Crossover = False
    Elitism = True
    Tournament = True
    SelectionStrength = 3
    PercentElitism = 0.1

    def __init__(self, individual):
        if isinstance(individual, list):
            self.set = individual
        elif isinstance(individual, Individual):
            self.set = Population.mutate(individual)
        else:
            self.set = Population.generate_random_population()

    def find_optimal(self, pred):
        score, optimal = -1, None
        for p in self.set:
            if pred.predict(p) > score:
                optimal = p
                score = pred.predict(p)
        return optimal

    def evolve(self, pred):
        parents = shuffle(self.selectParents(pred))
        new_population = []
        if Population.Elitism:
            nb_selected = int(Population.PercentElitism * Population.Size)
            sorted_set = Population.sort(self.set, pred)
            for i in xrange(nb_selected):
                new_population.append(sorted_set[i])
        tp = ThreadPool()
        tp.start()
        for i in parents:
            father = parents.pop()
            mother = parents.pop()
            tp.add(generate_children(father, mother, new_population, pred))
        tp.join()
        self.set = new_population
        return self

    def select_parents(self, pred):
        sorted_set = Population.sort(self.set, pred)
        parents = []
        if not Population.TruncateProportionalSelection:
            return self.set
        roulette = Population.create_roulette(sorted_set, pred)
        for i in xrange(Population.Size):
            parents.append(
                roulette[GeneticAlgorithm.get_random_int(self.set.length)]
            )
        return parents
