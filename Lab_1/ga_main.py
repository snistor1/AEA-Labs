import sys
import argparse
import time
import copy
import itertools
import logging
import multiprocessing as mproc
from abc import ABC
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skopt.callbacks import EarlyStopper
from skopt.utils import use_named_args
from skopt.space import Integer, Real
from skopt import gp_minimize

from selection.strategies import RankBased, Roulette, SUS
from evaluation.evaluation_cy import evaluate
from operators.operators_cy import upgrade_net, upgrade_input
from utils import PlotContext
from utils import collect, setup_logging
from ga_defines import *

NETWORK_SIZE = 9

NET_MUTATION_P = 0.05
NET_CROSSOVER_P = 0.6
NET_STRATEGY = SUS(n_elitism=N_ELITISM)
NET_POP_SIZE = 10000

INPUT_MUTATION_P = 0.05
INPUT_CROSSOVER_P = 0.6
INPUT_STRATEGY = SUS(n_elitism=12)

search_space = [Real(0.05, 1.0, name='net_mutation_p'), Real(0.05, 1.0, name='net_crossover_p'),
                Real(0.05, 1.0, name='input_mutation_p'), Real(0.05, 1.0, name='input_crossover_p'),
                Integer(1, 3, name='net_selection'), Integer(1, 3, name='input_selection'),
                Integer(100, 10000, name='net_pop_size')]


plot_config = {
    'net_mean': ('Mean network fitness (relative)', 'blue', '-'),
    'test_mean': ('Mean network fitness (absolute)', 'green', '-'),
    'global_max_rel': ('Global best network fitness (relative)', 'red', '--'),
    'global_max_abs': ('Global best network fitness (absolute)', 'darkorange', '--')
}


class Stopper(EarlyStopper, ABC):

    def __call__(self, result):
        return result.fun == 0.0


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
    )[:, :2].reshape((networks_shape)).astype(np.int64)
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
    return rng.integers(0, 2, size=size, dtype=np.int64) \
              .reshape((INPUT_POP_SIZE, -1))


def initialize_all_inputs():
    """
    Carteian product of {0,1} ^ NETWORK_SIZE
    """
    domain = np.repeat(np.array([[0, 1]]), NETWORK_SIZE, axis=0)
    return np.array(
        np.meshgrid(*domain)
    ).T.reshape(-1, NETWORK_SIZE).astype(np.int64)


def get_best_values(net_population, net_fitness, test_fitness=None):
    rel_best_idx = np.argmax(net_fitness)
    rel_val, rel_individual = (net_fitness[rel_best_idx],
                               net_population[rel_best_idx])
    if test_fitness is not None:
        abs_best_idx = np.argmax(test_fitness)
        abs_val, abs_individual = (test_fitness[abs_best_idx],
                                   net_population[abs_best_idx])
        return ((rel_val, rel_individual), (abs_val, abs_individual))
    return ((rel_val, rel_individual), (None, None))


def selection(population, fitness_values,
              strategy = RankBased()):
    return strategy(population, fitness_values)


def get_plot_context(active_dict):
    n_plots = sum(active_dict.values())
    labels, colors, line_styles = zip(*[plot_config[k]
                                       for k in active_dict
                                       if active_dict[k]])
    return PlotContext(n_plots,
                       title='Network fitness',
                       xlabel='Iterations',
                       ylabel='Fitness',
                       labels=labels,
                       colors=colors,
                       line_styles=line_styles)


def run_ga_plot(args):
    # set variables
    setup_logging()
    use_test_data = args.absolute or args.global_absolute
    active_plots = {
        'net_mean': args.relative,
        'test_mean': args.absolute,
        'global_max_rel': args.global_relative,
        'global_max_abs': args.global_absolute,
    }
    logger = logging.getLogger('general')
    plt_context = get_plot_context(active_plots)
    fitness_stats = pd.DataFrame(columns=['net_mean', 'net_std',
                                          'input_mean', 'input_std',
                                          'test_mean', 'test_std',
                                          'global_max_rel', 'global_max_abs'])
    start_time = time.time()
    # initialize populations
    net_population = initialize_networks(3, 20)
    input_population = initialize_inputs()
    test_input_population = initialize_all_inputs()
    # go through iteration 0
    net_fitness, input_fitness = evaluate(net_population, input_population)
    test_net_fitness = (evaluate(net_population, test_input_population)[0]
                        if use_test_data
                        else None)
    ((rel_val, rel_net),
     (abs_val, abs_net)) = get_best_values(net_population, net_fitness,
                                           test_fitness=test_net_fitness)
    fitness_stats = fitness_stats.append(
        collect(net_fitness, input_fitness, test_fit=test_net_fitness,
                global_abs=abs_val, global_rel=rel_val,
                collector=fitness_stats),
        ignore_index=True
    )
    plt_context.plot(0, (fitness_stats[k]
                         for k in active_plots
                         if active_plots[k]))
    # main loop 1..N_EPOCHS
    for i in range(1, N_EPOCHS+1):
        print(f'Epoch {i}/{N_EPOCHS}')
        logger.info('Current epoch: %d', i)
        net_population = selection(net_population, net_fitness, strategy=NET_STRATEGY)
        input_population = selection(input_population, input_fitness,
                                     strategy=INPUT_STRATEGY)
        upgrade_net(net_population, NETWORK_SIZE, NET_MUTATION_P, NET_CROSSOVER_P)
        upgrade_input(input_population, INPUT_MUTATION_P, INPUT_CROSSOVER_P)
        net_fitness, input_fitness = evaluate(net_population, input_population)
        test_net_fitness = (evaluate(net_population, test_input_population)[0]
                            if use_test_data
                            else None)
        ((new_rel_val, new_rel_net),
        (new_abs_val, new_abs_net)) = get_best_values(net_population, net_fitness,
                                                      test_fitness=test_net_fitness)
        if new_rel_val > rel_val:
            rel_val = new_rel_val
            rel_net = new_rel_net
        if test_net_fitness is not None and new_abs_val > abs_val:
            abs_val = new_abs_val
            abs_net = new_abs_net
        fitness_stats = fitness_stats.append(
            collect(net_fitness, input_fitness, test_fit=test_net_fitness,
                    global_abs=abs_val, global_rel=rel_val,
                    collector=fitness_stats),
            ignore_index=True
        )
        plt_context.plot(i, (fitness_stats[k]
                             for k in active_plots
                             if active_plots[k]))
    logger.info('Time elapsed: %f', time.time() - start_time)
    logger.info('Best network: %s', rel_net)
    plt.waitforbuttonpress()


@use_named_args(search_space)
def run_ga_opt(**params):
    global NET_MUTATION_P, NET_CROSSOVER_P, INPUT_MUTATION_P, INPUT_CROSSOVER_P
    global NET_STRATEGY, INPUT_STRATEGY, NET_POP_SIZE

    file = open('auto_tune.txt', 'a')
    print(params.values())
    print(params.values(), file=file)
    NET_MUTATION_P = params['net_mutation_p']
    NET_CROSSOVER_P = params['net_crossover_p']
    INPUT_MUTATION_P = params['input_mutation_p']
    INPUT_CROSSOVER_P = params['input_crossover_p']
    NET_POP_SIZE = params['net_pop_size']
    if params['net_selection'] == 1:
        NET_STRATEGY = RankBased()
    elif params['net_selection'] == 2:
        NET_STRATEGY = SUS(n_elitism=int(0.05 * NET_POP_SIZE))
    else:
        NET_STRATEGY = Roulette()
    if params['input_selection'] == 1:
        INPUT_STRATEGY = RankBased()
    elif params['input_selection'] == 2:
        INPUT_STRATEGY = SUS(n_elitism=12)
    else:
        INPUT_STRATEGY = Roulette()

    start_time = time.time()
    # initialize populations
    net_population = initialize_networks(3, 20)
    input_population = initialize_inputs()
    test_input_population = initialize_all_inputs()
    # go through iteration 0
    net_fitness, input_fitness = evaluate(net_population, input_population)
    test_net_fitness = (evaluate(net_population, test_input_population)[0])
    ((rel_val, rel_net),
     (abs_val, abs_net)) = get_best_values(net_population, net_fitness,
                                           test_fitness=test_net_fitness)
    # main loop 1..N_EPOCHS
    for i in range(1, N_EPOCHS + 1):
        net_population = selection(net_population, net_fitness, strategy=NET_STRATEGY)
        input_population = selection(input_population, input_fitness,
                                     strategy=INPUT_STRATEGY)
        upgrade_net(net_population, NETWORK_SIZE, NET_MUTATION_P, NET_CROSSOVER_P)
        upgrade_input(input_population, INPUT_MUTATION_P, INPUT_CROSSOVER_P)
        net_fitness, input_fitness = evaluate(net_population, input_population)
        test_net_fitness = (evaluate(net_population, test_input_population)[0])
        ((new_rel_val, new_rel_net),
         (new_abs_val, new_abs_net)) = get_best_values(net_population, net_fitness,
                                                       test_fitness=test_net_fitness)
        if new_rel_val > rel_val:
            rel_val = new_rel_val
            rel_net = new_rel_net
        if test_net_fitness is not None and new_abs_val > abs_val:
            abs_val = new_abs_val
            abs_net = new_abs_net
    print(f'RelVal: {rel_val}\nAbsVal: {abs_val}', file=file)
    print(f'Time elapsed: {time.time() - start_time}', file=file)
    print(f'Network: {rel_net}', file=file)
    file.close()
    return 1 - ((rel_val + abs_val) / 2)


def run_ga_exp(args):
    setup_logging()
    # set variables
    use_test_data = args.use_test_data
    logger = logging.getLogger('general')
    fitness_stats = pd.DataFrame(columns=['net_mean', 'net_std',
                                          'input_mean', 'input_std',
                                          'test_mean', 'test_std',
                                          'global_max_rel', 'global_max_abs'])
    # outer loop
    for r in range(args.repeats):
        logger.info('Starting run %d/%d...', r+1, args.repeats)
        print('Starting run %d/%d...' % (r+1, args.repeats))
        start_time = time.time()
        # initialize populations
        net_population = initialize_networks(3, 20)
        input_population = initialize_inputs()
        test_input_population = initialize_all_inputs()
        # go through iteration 0
        net_fitness, input_fitness = evaluate(net_population, input_population)
        test_net_fitness = (evaluate(net_population, test_input_population)[0]
                            if use_test_data
                            else None)
        ((rel_val, rel_net),
        (abs_val, abs_net)) = get_best_values(net_population, net_fitness,
                                              test_fitness=test_net_fitness)
        # main loop 1..N_EPOCHS
        for i in range(1, N_EPOCHS+1):
            net_population = selection(net_population, net_fitness, strategy=NET_STRATEGY)
            input_population = selection(input_population, input_fitness,
                                         strategy=INPUT_STRATEGY)
            upgrade_net(net_population, NETWORK_SIZE, NET_MUTATION_P, NET_CROSSOVER_P)
            upgrade_input(input_population, INPUT_MUTATION_P, INPUT_CROSSOVER_P)
            net_fitness, input_fitness = evaluate(net_population, input_population)
            test_net_fitness = (evaluate(net_population, test_input_population)[0]
                                if use_test_data
                                else None)
            ((new_rel_val, new_rel_net),
            (new_abs_val, new_abs_net)) = get_best_values(net_population, net_fitness,
                                                          test_fitness=test_net_fitness)
            if new_rel_val > rel_val:
                rel_val = new_rel_val
                rel_net = new_rel_net
            if test_net_fitness is not None and new_abs_val > abs_val:
                abs_val = new_abs_val
                abs_net = new_abs_net
        fitness_stats = fitness_stats.append(
            collect(net_fitness, input_fitness, test_fit=test_net_fitness,
                    global_abs=abs_val, global_rel=rel_val,
                    collector=fitness_stats),
            ignore_index=True
        )
        logger.info('Time elapsed: %f', time.time() - start_time)
        logger.info('Best network: %s', rel_net)
    fitness_stats.to_csv(args.out if args.out is not None else sys.stdout)


def auto_optimize(args):
    result = gp_minimize(run_ga_opt, search_space, callback=[Stopper()])
    print(f'Best Fitness: {result.fun}')
    print(f'Best params: {result.x}')


def parse_opts():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='allowed commands')
    parser_plot = subparsers.add_parser('plot', help='plot GA iterations')
    parser_exp = subparsers.add_parser('exp', help='run multiple experiments')
    parser_auto_hyper = subparsers.add_parser('auto_hyper', help='auto optimize hyper-parameters')

    parser.set_defaults(func=lambda x: parser.print_usage())

    parser_plot.add_argument('-a', '--absolute', action='store_true',
                        help='absolute mean fitness')
    parser_plot.add_argument('-r', '--relative', action='store_true',
                        help='relative mean fitness')
    parser_plot.add_argument('-g', '--global-absolute', action='store_true',
                        help='absolute best value')
    parser_plot.add_argument('-l', '--global-relative', action='store_true',
                        help='relative best value')
    parser_plot.set_defaults(func=run_ga_plot)

    parser_exp.add_argument('-r', action='store', metavar='<repeats>',
                            dest='repeats', type=int,
                            help='number of times to run the algorithm')
    parser_exp.add_argument('-o', action='store', metavar='<file>',
                            dest='out', type=str,
                            help='''name of output file; if not present, results
                            will be printed to standard output''')
    parser_exp.add_argument('-t', '--use-test-data', action='store_true',
                            help='use all possible inputs instead of a subset')
    parser_exp.set_defaults(func=run_ga_exp)
    parser_auto_hyper.set_defaults(func=auto_optimize)
    return parser.parse_args()


def main():
    args = parse_opts()
    args.func(args)


if __name__ == '__main__':
    main()

