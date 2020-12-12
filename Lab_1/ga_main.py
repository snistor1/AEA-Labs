import time
import copy
import itertools
import logging
import multiprocessing as mproc
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from selection.strategies import Strategy, RankBased, Roulette, SUS
from evaluation.evaluation_cy import evaluate
from operators.operators_cy import upgrade_net, upgrade_input
from utils import PlotContext
from utils import collect_and_log_fitness, setup_logging
from ga_defines import *


def initialize_networks(min_c=MIN_COMPARATORS, max_c=MAX_COMPARATORS):
    """
    Generate a random population of sorting networks, each with at least
    min_c comparators and at most max_c comparators.
    """
    if max_c > MAX_COMPARATORS:
        raise ValueError("max_c cannot exceed MAX_COMPARATORS")
    if min_c < 0:
        raise ValueError("min_c must be nonnegative")
    if min_c > max_c:
        raise ValueError("min_c cannot be greater than max_c")
    rng = np.random.default_rng(seed=0)
    # MAX_COMPARATORS + 1 sentinel value to mark the end
    networks_shape = (NET_POP_SIZE, MAX_COMPARATORS + 1, 2)
    # First, generate random pairs of row-distinct integers in
    # [0, NETWORK_SIZE). Since matrices must be rectangular (no jagged arrays),
    # we fix the maximum capacity for comparators to MAX_COMPARATORS.
    networks = np.argpartition(
        rng.random((NET_POP_SIZE * (MAX_COMPARATORS + 1), NETWORK_SIZE)),
        2,
        axis=-1
    )[:, :2].reshape((networks_shape))
    # Randomly generate the number of comparators each individual should have,
    # which is in the interval [min_c, max_c].
    n_comparators = rng.integers(min_c,
                                 max_c + 1,
                                 size=NET_POP_SIZE)
    # Generate a mask that tells us which comparators we should keep.
    mask = np.repeat(
        np.arange(MAX_COMPARATORS + 1)[..., np.newaxis],
        2,
        axis=-1
    ) < n_comparators.reshape(-1, 1, 1)
    # Mark the comparators we want to delete with a placeholder value. Since
    # comparators are pairs of indices and indices are always nonnegative, -1
    # is a good choice.
    networks[~mask] = -1
    # Sort each comparator s.t. they are of the form (a, b) where a <= b.
    return np.ascontiguousarray(np.sort(networks))


def initialize_inputs():
    """
    Generate a random population of binary inputs.
    """
    rng = np.random.default_rng(seed=0)
    size = INPUT_POP_SIZE * NETWORK_SIZE
    return rng.integers(0, 2, size=size) \
              .reshape((INPUT_POP_SIZE, -1))


def initialize_all_inputs():
    """
    Carteian product of {0,1} ^ NETWORK_SIZE
    """
    domain = np.repeat(np.array([[0, 1]]), NETWORK_SIZE, axis=0)
    return np.array(
        np.meshgrid(*domain)
    ).T.reshape(-1, NETWORK_SIZE)


def get_best_individual(population, fitness_values):
    local_best = np.argmax(fitness_values)
    best_val = fitness_values[local_best]
    best_individual = population[local_best]
    return best_val, best_individual


def selection(population, fitness_values,
              strategy = RankBased()):
    return strategy(population, fitness_values)


def main():
    setup_logging(log_level=logging.INFO)
    context = PlotContext(2,
                          title='Network fitness (relative)',
                          xlabel='Iterations',
                          ylabel='Fitness',
                          labels=['Mean network fitness (relative)',
                                  # 'Mean network fitness (absolute)',
                                  'Global best network fitness (relative)'],
                                  # 'Global best network fitness (absolute)'],
                          colors=['blue', 'red'],
                          line_styles=['-', '-'])
    logger = logging.getLogger('general')
    columns = ['net_max', 'net_mean', 'net_std',
               'input_max', 'input_mean', 'input_std']
               # 'test_max', 'test_mean', 'test_std']
    fitness_stats = pd.DataFrame(columns=columns)
    global_max_rel = []
    global_max_abs = []
    net_strategy = RankBased(0.995)
    input_strategy = Roulette()

    start_time = time.time()
    population = initialize_networks(40, 60)
    input_population = initialize_inputs()
    test_input_population = initialize_all_inputs()
    net_fitness, input_fitness = evaluate(population, input_population)
    # test_net_fitness, _ = evaluate(population, test_input_population)
    collect_and_log_fitness(net_fitness, input_fitness,
                            # test_population_fit=test_net_fitness,
                            collector=fitness_stats)
    best_val, best_individual = get_best_individual(population, net_fitness)
    best_input_val, best_input_individual = get_best_individual(input_population, input_fitness)
    # global_max_abs_fitness, _ = evaluate([best_individual], test_input_population)
    global_max_rel.append(best_val)
    # global_max_abs.append(global_max_abs_fitness[0])
    context.plot(0, (fitness_stats['net_mean'],
                     # fitness_stats['test_mean'],
                     global_max_rel
                     # global_max_abs
                     ))
    for i in range(1, N_EPOCHS+1):
        print(f'Epoch {i}/{N_EPOCHS}')
        logger.info('Current epoch: %d', i)
        population = selection(population, net_fitness, strategy=net_strategy)
        input_population = selection(input_population, input_fitness,
                                     strategy=input_strategy)
        upgrade_net(population)
        upgrade_input(input_population)
        net_fitness, input_fitness = evaluate(population, input_population)
        # test_net_fitness, _ = evaluate(population, test_input_population)
        collect_and_log_fitness(net_fitness, input_fitness,
                                # test_population_fit=test_net_fitness,
                                collector=fitness_stats)
        new_best_val, new_best_individual = get_best_individual(population, net_fitness)
        new_best_input_val, new_best_input_individual = get_best_individual(input_population, input_fitness)
        if new_best_val > best_val:
            best_val = new_best_val
            best_individual = new_best_individual
        if new_best_input_val > best_input_val:
            best_input_val = new_best_input_val
            best_input_individual = new_best_input_individual
        # global_max_abs_fitness, _ = evaluate([best_individual], test_input_population)
        global_max_rel.append(best_val)
        # global_max_abs.append(global_max_abs_fitness[0])
        context.plot(i, (fitness_stats['net_mean'],
                         # fitness_stats['test_mean'],
                         global_max_rel
                         # global_max_abs
                         ))
    logger.info('Time elapsed: %f', time.time() - start_time)
    logger.info('Best network: %s', best_individual)
    logger.debug('Input population:')
    logging.getLogger('general.array').debug(pformat(input_population))
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

