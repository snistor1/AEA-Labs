import time
import copy
import itertools
import logging
import multiprocessing as mproc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from selection.strategies import Strategy, RankBased, Roulette
from evaluation import evaluate
from operators import upgrade, upgrade_input
from utils import PlotContext
from utils import collect_and_log_fitness, setup_logging
from ga_defines import *


def generate_population() -> list:
    population = list()
    for _ in range(NET_POP_SIZE):
        nr_comparators = np.random.randint(3, 20)
        network = list()
        for _ in range(nr_comparators):
            first = np.random.randint(NETWORK_SIZE)
            second = np.random.choice([i for i in range(NETWORK_SIZE) if i != first])
            if first > second:
                first, second = second, first
            network.append((first, second))
        population.append(network)
    return population


def generate_input_population() -> list:
    return [np.random.randint(2, size=NETWORK_SIZE) for _ in range(IN_POP_SIZE)]


def get_best_individual(population, fitness_values) -> (float, list):
    local_best = np.argmax(fitness_values)
    best_val = fitness_values[local_best]
    best_individual = population[local_best]
    return best_val, best_individual


def selection(population: list, fitness_values: np.ndarray,
              strategy: Strategy = RankBased()) -> list:
    return strategy(population, fitness_values)


def generate_all_inputs():
    # # Carteian product of {0,1} ^ NETWORK_SIZE
    # domain = np.repeat(np.array([[0, 1]]), NETWORK_SIZE, axis=0)
    # return np.array(
    #     np.meshgrid(*domain)
    # ).T.reshape(-1, NETWORK_SIZE)
    return [np.array(tup) for tup in itertools.product(*[[0, 1] for _ in range(NETWORK_SIZE)])]


def main():
    setup_logging(log_level=logging.DEBUG)
    context = PlotContext(2,
                          title='Mean network fitness (relative and absolute)',
                          labels=['Mean relative network fitness',
                                  'Mean absolute network fitness'],
                          colors=['blue', 'green'],
                          line_styles=['--', '--'])
    logger = logging.getLogger('general')
    columns = ['net_max', 'net_mean', 'net_std',
               'input_max', 'input_mean', 'input_std',
               'test_max', 'test_mean', 'test_std']
    fitness_stats = pd.DataFrame(columns=columns)

    start_time = time.time()
    population = generate_population()
    input_population = generate_input_population()
    test_input_population = generate_all_inputs()
    net_fitness, input_fitness = evaluate(population, input_population,
                                          multiprocessing=True)
    test_net_fitness, _ = evaluate(population, test_input_population,
                                    multiprocessing=True)
    collect_and_log_fitness(net_fitness, input_fitness,
                            test_population_fit=test_net_fitness,
                            collector=fitness_stats)
    best_val, best_individual = get_best_individual(population, net_fitness)
    best_input_val, best_input_individual = get_best_individual(input_population, input_fitness)
    context.plot(0, (fitness_stats['net_mean'], fitness_stats['test_mean']))
    for i in range(1, N_EPOCHS+1):
        print(f'Epoch {i}/{N_EPOCHS}')
        logger.info('Current epoch: %d', i)
        population = selection(population, net_fitness, strategy=Roulette())
        input_population = selection(input_population, input_fitness,
                                     strategy=Roulette())
        population = upgrade(population)
        input_population = upgrade_input(input_population)
        net_fitness, input_fitness = evaluate(population, input_population,
                                              multiprocessing=True)
        test_net_fitness, _ = evaluate(population, test_input_population,
                                       multiprocessing=True)
        collect_and_log_fitness(net_fitness, input_fitness,
                                test_population_fit=test_net_fitness,
                                collector=fitness_stats)
        new_best_val, new_best_individual = get_best_individual(population, net_fitness)
        new_best_input_val, new_best_input_individual = get_best_individual(input_population, input_fitness)
        context.plot(i, (fitness_stats['net_mean'], fitness_stats['test_mean']))
        if new_best_val > best_val:
            best_val = new_best_val
            best_individual = new_best_individual
        if new_best_input_val > best_input_val:
            best_input_val = new_best_input_val
            best_input_individual = new_best_input_individual
    logger.info('Time elapsed: %f', time.time() - start_time)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

