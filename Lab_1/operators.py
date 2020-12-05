import copy

import numpy as np

from ga_defines import *


def mutation_indices(n_genes: int, pm: float, K: int = 2) -> np.ndarray:
    idxs = np.cumsum(
        np.ceil(
            np.log(
                np.random.rand(np.ceil(K * pm * n_genes).astype(int))
            )
            /
            np.log(1 - pm)
        ).astype(int)
    )
    return idxs[:np.searchsorted(idxs, n_genes)]


def upgrade(population: list) -> list:
    new_population = list()
    probabilities = np.cumsum(NET_MUTATION_CASES)
    for individual in population:
        r = np.random.rand()
        if r < NET_MUTATION_P:
            mutation_procedure = np.random.rand()
            selected = np.searchsorted(probabilities, mutation_procedure)
            if selected == 0:
                first = np.random.randint(len(individual))
                second = np.random.choice([i for i in range(len(individual)) if i != first])
                f0 = individual[first][0]
                f1 = individual[first][1]
                s0 = individual[second][0]
                s1 = individual[second][1]
                individual[first] = (s0, f1) if s0 <= f1 else (f1, s0)
                individual[second] = (f0, s1) if f0 <= s1 else (s1, f0)
            elif selected == 1:
                first = np.random.randint(NETWORK_SIZE)
                second = np.random.choice([i for i in range(NETWORK_SIZE) if i != first])
                if first > second:
                    first, second = second, first
                index_to_add = np.random.randint(len(individual) + 1)
                individual.insert(index_to_add, (first, second))
            else:
                if len(individual) > 3:
                    to_remove = np.random.randint(len(individual))
                    del individual[to_remove]
        new_population.append(copy.deepcopy(individual))

    crossover_indices = [i for i in range(len(new_population)) if np.random.rand() < NET_CROSSOVER_P]
    if len(crossover_indices) % 2 != 0:
        del crossover_indices[-1]
    for i in range(0, len(crossover_indices), 2):
        idx_1, idx_2 = crossover_indices[i], crossover_indices[i + 1]
        cut_point = np.random.randint(1, min(len(new_population[idx_1]), len(new_population[idx_2])))
        new_ind_1 = new_population[idx_1][:cut_point] + new_population[idx_2][cut_point:]
        new_ind_2 = new_population[idx_2][:cut_point] + new_population[idx_1][cut_point:]
        new_population[idx_1], new_population[idx_2] = copy.deepcopy(new_ind_1), copy.deepcopy(new_ind_2)
    return new_population


def upgrade_input(input_population: list) -> list:
    new_population = list()
    for individual in input_population:
        new_population.append(np.where(np.random.rand(NETWORK_SIZE) < IN_MUTATION_P, 1 - individual, individual))

    crossover_indices = [i for i in range(len(new_population)) if np.random.rand() < IN_CROSSOVER_P]
    if len(crossover_indices) % 2 != 0:
        del crossover_indices[-1]
    for i in range(0, len(crossover_indices), 2):
        idx_1, idx_2 = crossover_indices[i], crossover_indices[i + 1]
        cut_point = NETWORK_SIZE // 2
        new_ind_1 = np.concatenate([new_population[idx_1][:cut_point], new_population[idx_2][cut_point:]])
        new_ind_2 = np.concatenate([new_population[idx_2][:cut_point], new_population[idx_1][cut_point:]])
        new_population[idx_1], new_population[idx_2] = copy.deepcopy(new_ind_1), copy.deepcopy(new_ind_2)
    return new_population
