from abc import ABC, abstractmethod

import numpy as np


class Strategy(ABC):
    def __init__(self, n_elitism: int = 0):
        self.n_elitism = n_elitism

    @abstractmethod
    def process(self, raw_fitness: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def select(self, population: list,
               fitness: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, population: list,
                 raw_fitness: np.ndarray) -> list:
        return self.select(population, self.process(raw_fitness))

class RankBased(Strategy):
    def __init__(self, base: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def process(self, raw_fitness: np.ndarray) -> np.ndarray:
        # Convert fitness values based on rank
        sorted_indices = np.argsort(raw_fitness)[::-1]
        inverse_perm = np.argsort(sorted_indices)
        new_fitness = self.base ** np.arange(sorted_indices.size)
        new_fitness = new_fitness[inverse_perm]
        return new_fitness

    def select(self, population: list,
                 fitness: np.ndarray) -> list:
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cumulative_probabilities = np.cumsum(individual_probabilities)
        new_population = []
        if self.n_elitism > 0:
            sorted_indices = np.argsort(fitness)[::-1]
            chosen_elitism_values = sorted_indices[:self.n_elitism]
            new_population.extend(
                [population[idx] for idx in chosen_elitism_values]
            )
        r = np.random.rand(population_size - self.n_elitism)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population.extend([population[idx] for idx in selected])
        return new_population

class Roulette(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, raw_fitness: np.ndarray) -> np.ndarray:
        return raw_fitness

    def select(self, population: list,
               fitness: np.ndarray) -> list:
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cumulative_probabilities = np.cumsum(individual_probabilities)
        new_population = []
        if self.n_elitism > 0:
            sorted_indices = np.argsort(fitness)[::-1]
            chosen_elitism_values = sorted_indices[:self.n_elitism]
            new_population.extend(
                [population[idx] for idx in chosen_elitism_values]
            )
        r = np.random.rand(population_size - self.n_elitism)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population.extend([population[idx] for idx in selected])
        return new_population


class SUS(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, raw_fitness: np.ndarray) -> np.ndarray:
        return raw_fitness

    def select(self, population: list,
               fitness: np.ndarray) -> list:
        population_size = fitness.size
        total_fitness = np.sum(fitness)
        individual_probabilities = fitness / total_fitness
        cummulative_probabilites = np.cumsum(individual_probabilities)
        new_population = []
        if self.n_elitism > 0:
            sorted_indices = np.argsort(fitness)[::-1]
            chosen_elitism_values = sorted_indices[:self.n_elitism]
            new_population.extend(
                [population[idx] for idx in chosen_elitism_values]
            )
        dist = 1.0 / population_size
        r = np.random.rand() / population_size
        pointers = r + dist * np.arange(population_size)
        selected = np.searchsorted(cummulative_probabilites, pointers)
        new_population.extend([population[idx] for idx in selected])
        return new_population

