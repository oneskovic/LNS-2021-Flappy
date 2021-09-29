from organism import Organism
from typing import List
import numpy as np
from evaluator import Evaluator
import operator
import bisect
from copy import deepcopy
import torch


class GeneticAlgorithm:
    def __init__(self, hparams, evaluator):
        self.hparams = hparams
        self.population: List[Organism] = list()
        self.evaluator = evaluator
        self.init_population()

    def init_population(self):
        pop_size = self.hparams['population_size']
        self.population = [Organism(self.hparams) for _ in range(pop_size)]
        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)

    def __crossover(self, organism1: Organism, organism2: Organism):
        child = deepcopy(organism1)
        # for i in range(len(organism1.weights)):
        #     w_bin_mat = torch.rand_like(organism1.weights[i]) <= 0.5
        #     b_bin_mat = torch.rand_like(organism1.biases[i]) <= 0.5
        #
        #     w = organism1.weights[i] * w_bin_mat + organism2.weights[i] * (~w_bin_mat)
        #     b = organism1.biases[i] * b_bin_mat + organism2.biases[i] * (~b_bin_mat)
        #     # child.weights[i] = (organism1.weights[i] + organism2.weights[i])/2.0
        #     # child.biases[i] = (organism1.biases[i] + organism2.biases[i])/2.0
        #     child.weights[i] = w
        #     child.biases[i] = b
        # child.update_model_params()
        return child

    def reproduce(self, offspring_cnt):
        tournament_size = self.hparams['tournament_size']
        parent1 = np.random.randint(0,len(self.population),(tournament_size,offspring_cnt)).min(axis=0)
        parent2 = np.random.randint(0, len(self.population), (tournament_size, offspring_cnt)).min(axis=0)
        children = [self.__crossover(self.population[parent1[i]],self.population[parent2[i]]) for i in range(offspring_cnt)]
        for i in range(offspring_cnt):
            children[i].mutate()
        return children

    def tournament2(self):
        new_population = []
        n = len(self.population)*1.0
        probs = np.array([(n-i)/n for i in range(int(n))])
        rand = np.random.uniform(0,1,int(n))
        should_pick = (rand <= probs)
        for i in range(int(n)):
            if should_pick[i]:
                new_population.append(self.population[i])
        return new_population

    def evaluate_population(self):
        for i in range(len(self.population)):
            self.population[i].score = self.evaluator.evaluate(self.population[i], False)

    def step(self):
        elite_cnt = int(self.hparams['elite_fraction'] * len(self.population))
        remove_cnt = int(self.hparams['remove_fraction'] * len(self.population))
        new_population = self.population[:elite_cnt]
        del self.population[-remove_cnt:]
        new_population += self.reproduce(self.hparams['population_size'] - elite_cnt)
        self.population = new_population
        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)