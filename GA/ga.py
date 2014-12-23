# -*- coding: utf-8 -*-

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, randint
from ThreadPool.threadpool import ThreadPool
from threading import Lock

GOAL = 0.90
INDIVIDUAL_SIZE = 784
POPULATION_SIZE = 150


class GeneticAlgorithm():
    def __init__(self, pred, ind=None):
        self.init_plot()
        self.predictor = pred
        self.population = Population(ind)
        self.optimal_individual = ind if ind else Individual()

    def find_optimal(self):
        self.evolution = []
        generation = 0
        while self.predictor.predict(self.optimal_individual) < GOAL:
            self.population = self.population.evolve(self.predictor)
            self.optimal_individual = self.population.find_optimal(
                self.predictor
            )
            opt = self.predictor.predict(self.optimal_individual)
            self.evolution.append(opt)
            self.plot()
            print 'Generation ' + str(generation) + ', best score: ' + str(opt)
            generation += 1
        return self.optimal_individual

    def init_plot(self):
        plt.ion()
        self.graph, = plt.plot([], [], 'ro')
        plt.xlabel('Generation')
        plt.ylabel('Optimal score')
        plt.show()

    def plot(self):
        pass
        self.graph.set_xdata(self.evolution)
        self.graph.set_ydata([i for i in xrange(len(self.evolution))])
        plt.plot((np.array(self.evolution)*100).tolist())
        plt.draw()
        plt.show()

    @staticmethod
    def get_random_binary():
        return randint(0, 1)

    @staticmethod
    def get_random_float():
        return random.random()

    @staticmethod
    def get_random_int(limit):
        return int(random.uniform(0, limit))


class Individual():
    Size = INDIVIDUAL_SIZE

    def __init__(self, genes=None):
        if not genes:
            genes = [GeneticAlgorithm.get_random_binary()
                for i in xrange(Individual.Size)]
        self.genes = genes

    def random_mutate(self):
        for i in xrange(len(self.genes)):
            if GeneticAlgorithm.get_random_float() < 1.0/float(len(self.genes)):
                self.set_gene(i, GeneticAlgorithm.get_random_binary())
        return self

    def breed(self, parent):
        for i in xrange(len(self.genes)):
            self.set_gene(
                i,
                self.get_gene(i) if GeneticAlgorithm.get_random_binary == 1 else parent.get_gene(i)
            )
        return self

    def get_gene(self, place):
        return self.genes[place]

    def set_gene(self, place, gene):
        self.genes[place] = int(gene)

    def to_string(self):
        return ''.join(str(self.genes))


class Population():
    Lock = Lock()
    Size = POPULATION_SIZE
    TruncateProportionalSelection = True
    Crossover = False
    Elitism = True
    Tournament = True
    SelectionStrength = 3
    PercentElitism = 0.15

    def __init__(self, individual=None):
        if isinstance(individual, list):
            self.population = individual
        elif isinstance(individual, Individual):
            self.population = Population.mutate(individual)
        else:
            self.population = Population.generate_random_population()

    def find_optimal(self, pred):
        score, optimal = -1, None
        for p in self.population:
            if pred.predict(p) > score:
                optimal = p
                score = pred.predict(p)
        return optimal

    def evolve(self, pred):
        parents = self.select_parents(pred)
        shuffle(parents)
        new_population = []
        if Population.Elitism:
            nb_selected = int(Population.PercentElitism * Population.Size)
            sorted_set = Population.sort(self.population, pred)
            for i in xrange(nb_selected):
                new_population.append(sorted_set[i])
        tp = ThreadPool()
        tp.start()
        for i in parents:
            father = parents.pop()
            mother = parents.pop()
            task = lambda: Population.generate_children(father, mother, new_population, pred)
            tp.add(task)
        tp.join()
        self.population = new_population
        return self

    def select_parents(self, pred):
        sorted_set = Population.sort(self.population, pred)
        parents = []
        if not Population.TruncateProportionalSelection:
            return sorted_set
        roulette = Population.create_roulette(sorted_set, pred)
        for i in xrange(Population.Size):
            parents.append(
                roulette[GeneticAlgorithm.get_random_int(len(roulette) - 1)]
            )
        return parents

    @staticmethod
    def create_roulette(popu, pred):
        roulette = []
        fitnessTotal = sum(math.exp(pred.predict(x) *
                           Population.SelectionStrength) for x in popu)
        for ind in popu:
            to_create = 1 + math.exp(pred.predict(ind) * Population.SelectionStrength) / fitnessTotal
            for j in xrange(int(to_create)):
                roulette.append(ind)
        return roulette

    @staticmethod
    def sort(popu, pred):
        return sorted(
            popu,
            key=lambda x: pred.predict(x),
            reverse=True
        )

    @staticmethod
    def mutate(individual):
        population = [individual, ]
        for i in xrange(Population.Size - 1):
            i = copy.deepcopy(individual)
            population.append(i.random_mutate())

    @staticmethod
    def generate_random_population():
        return [Individual() for i in xrange(Population.Size)]

    @staticmethod
    def generate_children(father, mother, population, pred):
        son = copy.deepcopy(father)
        daughter = copy.deepcopy(mother)
        if Population.Crossover:
            son = son.breed(mother)
            daughter = daughter.breed(father)
        else:
            son.random_mutate()
            daughter.random_mutate()
        if Population.Tournament:
            sub = Population.sort([father, mother, son, daughter], pred)
            with Population.Lock:
                population.append(sub[0])
                population.append(sub[1])
        else:
            with Population.Lock:
                population.append(son, daughter)
        return population
