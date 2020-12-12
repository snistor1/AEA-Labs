from abc import ABC, abstractmethod

import numpy as np


class Strategy(ABC):
    def __init__(self, n_elitism=0):
        self.n_elitism = n_elitism

    @abstractmethod
    def process(self, raw_fitness):
        pass

    @abstractmethod
    def select(self, population, fitness):
        pass

    def __call__(self, population,
                 raw_fitness):
        return self.select(population, self.process(raw_fitness))

class RankBased(Strategy):
    def __init__(self, base=0.95, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def process(self, raw_fitness):
        # Convert fitness values based on rank
        sorted_indices = np.argsort(raw_fitness)[::-1]
        inverse_perm = np.argsort(sorted_indices)
        new_fitness = self.base ** np.arange(sorted_indices.size)
        new_fitness = new_fitness[inverse_perm]
        return new_fitness

    def select(self, population, fitness):
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cumulative_probabilities = np.cumsum(individual_probabilities)
        sorted_indices = np.argsort(fitness)[::-1]
        chosen_elitism_indices = sorted_indices[:self.n_elitism]
        r = np.random.rand(population_size - self.n_elitism)
        selected = np.concatenate((
            chosen_elitism_indices,
            np.searchsorted(cumulative_probabilities, r, side='left')
        ))
        return population[selected]

class Roulette(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, raw_fitness):
        return raw_fitness

    def select(self, population, fitness):
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cumulative_probabilities = np.cumsum(individual_probabilities)
        sorted_indices = np.argsort(fitness)[::-1]
        chosen_elitism_indices = sorted_indices[:self.n_elitism]
        r = np.random.rand(population_size - self.n_elitism)
        selected = np.concatenate((
            chosen_elitism_indices,
            np.searchsorted(cumulative_probabilities, r, side='left')
        ))
        return population[selected]


class SUS(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, raw_fitness):
        return raw_fitness

    def select(self, population, fitness):
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cummulative_probabilities = np.cumsum(individual_probabilities)
        sorted_indices = np.argsort(fitness)[::-1]
        chosen_elitism_indices = sorted_indices[:self.n_elitism]
        new_pop_size = population_size - self.n_elitism
        dist = 1.0 / new_pop_size
        r = np.random.rand() / new_pop_size
        pointers = r + dist * np.arange(new_pop_size)
        selected = np.concatenate((
            chosen_elitism_indices,
            np.searchsorted(cummulative_probabilities, pointers,
                            side='left')
        ))
        return population[selected]
