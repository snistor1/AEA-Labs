import sys
import time
import copy
import numpy as np

from ga_defines import *


def generate_population() -> list:
    population = list()
    for _ in range(POP_SIZE):
        nr_comparators = np.random.randint(60, 121)
        network = list()
        for _ in range(nr_comparators):
            first = np.random.randint(NETWORK_SIZE)
            second = np.random.choice([i for i in range(NETWORK_SIZE) if i != first])
            network.append((first, second))
        population.append(network)
    return population


def generate_input_population() -> list:
    return [np.random.randint(2, size=NETWORK_SIZE) for _ in range(INPUT_POP_SIZE)]


def eval_input(network, input_test_case) -> bool:
    for comparator in network:
        input_test_case[[comparator[0], comparator[1]]] = input_test_case[[comparator[1], comparator[0]]]
    return np.all(input_test_case[:-1] <= input_test_case[1:])


def fitness_networks(population: list, input_population: list) -> list:
    fitness_values = list()
    for network in population:
        fitness = sum([eval_input(network, test_case) for test_case in input_population]) / len(input_population)
        fitness_values.append(fitness)
    return fitness_values


def fitness_input(input_population: list, population: list) -> list:
    fitness_values = list()
    for test_case in input_population:
        fitness = len(population) / sum([eval_input(network, test_case) for network in population])
        fitness_values.append(fitness)
    return fitness_values


def get_best_individual(population, fitness_values) -> (float, list):
    local_best = np.argmax(fitness_values)
    best_val = fitness_values[local_best]
    best_individual = population[local_best]
    return best_val, best_individual


def upgrade(population: list) -> list:
    new_population = list()
    for individual in population:
        r = np.random.rand()
        if r < MUTATION_PROB:
            first = np.random.randint(len(individual))
            second = np.random.choice([i for i in range(len(individual)) if i != first])
            individual[first], individual[second] = individual[second], individual[first]
        new_population.append(individual)

    crossover_indices = [i for i in range(len(new_population)) if np.random.rand() < CROSS_OVER_PROB]
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
        new_population.append(np.where(np.random.rand(NETWORK_SIZE) < MUTATION_PROB, 1 - individual, individual))

    crossover_indices = [i for i in range(len(new_population)) if np.random.rand() < CROSS_OVER_PROB]
    if len(crossover_indices) % 2 != 0:
        del crossover_indices[-1]
    for i in range(0, len(crossover_indices), 2):
        idx_1, idx_2 = crossover_indices[i], crossover_indices[i + 1]
        cut_point = NETWORK_SIZE / 2
        new_ind_1 = new_population[idx_1][:cut_point] + new_population[idx_2][cut_point:]
        new_ind_2 = new_population[idx_2][:cut_point] + new_population[idx_1][cut_point:]
        new_population[idx_1], new_population[idx_2] = copy.deepcopy(new_ind_1), copy.deepcopy(new_ind_2)
    return new_population


def selection(population: list, fitness_values: list, elitism_nr=0, base=0.95) -> list:
    sorted_indices = np.argsort(fitness_values)[::-1]
    inverse_perm = np.argsort(sorted_indices)
    new_fitness = base ** np.arange(sorted_indices.size)
    new_fitness = new_fitness[inverse_perm]
    total_fitness = sum(new_fitness)
    individual_probabilities = [fitness_val / total_fitness for fitness_val in new_fitness]
    cumulative_probabilities = np.cumsum(individual_probabilities)
    if not elitism_nr:
        r = np.random.rand(POP_SIZE)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population = [population[idx] for idx in selected]
    else:
        best_fitness_values = sorted(fitness_values, reverse=True)[:elitism_nr]
        chosen_elitism_values = [np.where(fitness_values == i)[0][0] for i in best_fitness_values]
        r = np.random.rand(POP_SIZE - elitism_nr)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population = [population[idx] for idx in selected]
        new_population.extend([population[idx] for idx in chosen_elitism_values])
    return new_population


def main(log_to_file=True):
    start_time = time.time()
    population = generate_population()
    input_population = generate_input_population()
    fitness_values = fitness_networks(population, input_population)
    input_fitness_values = fitness_input(input_population, population)
    best_val, best_individual = get_best_individual(population, fitness_values)
    best_input_val, best_input_individual = get_best_individual(input_population, input_fitness_values)
    fp = open('log.txt', 'w') if log_to_file else sys.stdout
    for i in range(NR_EPOCHS):
        print(f'Current epoch: {i}', file=fp)
        population = selection(population, fitness_values, elitism_nr=ELITISM_NR)
        input_population = selection(input_population, input_fitness_values)
        population = upgrade(population)
        input_population = upgrade_input(input_population)
        fitness_values = fitness_networks(population, input_population)
        input_fitness_values = fitness_input(input_population, population)
        new_best_val, new_best_individual = get_best_individual(population, fitness_values)
        new_best_input_val, new_best_input_individual = get_best_individual(input_population, input_fitness_values)
        print(f'Current best: {best_val}', file=fp)
        print(f'New best: {new_best_val}', file=fp)
        print(f'Current input best: {best_input_val}', file=fp)
        print(f'New input best: {new_best_input_val}', file=fp)
        if new_best_val > best_val:
            best_val = new_best_val
            best_individual = new_best_individual
        if new_best_input_val > best_input_val:
            best_input_val = new_best_input_val
            best_input_individual = new_best_input_individual
    print(f'The best loss was {best_val}!', file=fp)
    print(f'Best individual: {best_individual}', file=fp)
    print(f'Best input individual: {best_input_individual}', file=fp)
    print(f'Time taken: {time.time() - start_time} seconds!', file=fp)


if __name__ == '__main__':
    main()
