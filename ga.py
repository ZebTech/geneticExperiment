# -*- coding: utf-8 -*-

import random
from random import shuffle
import copy
import ThreadPool
from threading import Lock

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
    Size = 64

    def __init__(self, genes):
        if not genes:
            genes = [GeneticAlgorithm.get_random_binary()
                for i in xrange(Individual.Size)]
        self.genes = genes

    def random_mutate(self):
        for i in xrange(len(self.genes)):
            if GeneticAlgorithm.get_random_float() < 1.0/(self.genes):
                self.set_gene(i, GeneticAlgorithm.get_random_binary())

    def breed(self, parent):
        for i in xrange(len(self.genes)):
            self.set_gene(
                i,
                self.get_gene() if GeneticAlgorithm.get_random_binary == 1 else parent.get_gene(i)
            )

    def get_gene(self, place):
        return self.set[place]

    def set_gene(self, place, gene):
        self.genes[place] = int(gene)

    def to_string(self):
        return self.genes.join('')



class Population():
    Lock = Lock()
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
            tp.add(Population.generate_children(
                father,
                mother,
                new_population,
                pred))
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

    @staticmethod
    def sort(set, pred):
        return set.sort(
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
            sub = Population.sort([father, mother, son, daughter])
            with Population.Lock:
                population.append(sub[0], sub[1])
        else:
            with Population.Lock:
                population.append(son, daughter)
