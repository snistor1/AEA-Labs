import time
import numpy as np

from ga_defines import *


def generate_population() -> list:
    pass


def generate_input_population() -> list:
    pass


def eval_input(network, input_test_case) -> int:
    pass


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
    pass


def upgrade_input(input_population: list) -> list:
    pass


def selection(population: list, fitness_values: list, elitism=False, base=0.95) -> list:
    sorted_indices = np.argsort(fitness_values)[::-1]
    inverse_perm = np.argsort(sorted_indices)
    new_fitness = base ** np.arange(sorted_indices.size)
    new_fitness = new_fitness[inverse_perm]
    total_fitness = sum(new_fitness)
    individual_probabilities = [fitness_val / total_fitness for fitness_val in new_fitness]
    cumulative_probabilities = np.cumsum(individual_probabilities)
    if not elitism:
        r = np.random.rand(POP_SIZE)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population = [population[idx] for idx in selected]
    else:
        elitism_nr = int(POP_SIZE * 0.1)
        best_fitness_values = sorted(fitness_values, reverse=True)[:elitism_nr]
        chosen_elitism_values = [np.where(fitness_values == i)[0][0] for i in best_fitness_values]
        r = np.random.rand(POP_SIZE - elitism_nr)
        selected = np.searchsorted(cumulative_probabilities, r)
        new_population = [population[idx] for idx in selected]
        new_population.extend([population[idx] for idx in chosen_elitism_values])
    return new_population


def main():
    start_time = time.time()
    population = generate_population()
    input_population = generate_input_population()
    fitness_values = fitness_networks(population, input_population)
    input_fitness_values = fitness_input(input_population, population)
    best_val, best_individual = get_best_individual(population, fitness_values)
    best_input_val, best_input_individual = get_best_individual(input_population, input_fitness_values)
    for i in range(NR_EPOCHS):
        print(f'Current epoch: {i}')
        population = selection(population, fitness_values, elitism=USE_ELITISM)
        input_population = selection(input_population, input_fitness_values, elitism=USE_ELITISM)
        population = upgrade(population)
        input_population = upgrade_input(input_population)
        fitness_values = fitness_networks(population, input_population)
        input_fitness_values = fitness_input(input_population, population)
        new_best_val, new_best_individual = get_best_individual(population, fitness_values)
        new_best_input_val, new_best_input_individual = get_best_individual(input_population, input_fitness_values)
        print(f'Current best: {best_val}')
        print(f'New best: {new_best_val}')
        print(f'Current input best: {best_input_val}')
        print(f'New input best: {new_best_input_val}')
        if new_best_val > best_val:
            best_val = new_best_val
            best_individual = new_best_individual
        if new_best_input_val > best_input_val:
            best_input_val = new_best_input_val
            best_input_individual = new_best_input_individual
    print(f'The best loss was {best_val}!')
    print(f'Time taken: {time.time() - start_time} seconds!')
    print(best_individual)


if __name__ == '__main__':
    main()
