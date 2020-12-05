import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ga_defines import *


def compare_swap(lst, i, j):
    if lst[i] > lst[j]:
        lst[i], lst[j] = lst[j], lst[i]


def sort(net, lst):
    for i, j in net:
        compare_swap(lst, i, j)


def outputs(net, n):
    sortees = [list(elem) for elem in itertools.product([0, 1], repeat=n)]
    for sortee in sortees:
        sort(net, sortee)
    return set(map(tuple, sortees))


class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self.__formatters = formatters
        self.__default_formatter = default_formatter

    def format(self, record):
        logger = logging.getLogger(record.name)
        while logger:
            if logger.name in self.__formatters:
                formatter = self.__formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # No formatter found, use default:
            formatter = self.__default_formatter
        return formatter.format(record)


class PlotContext:
    def __init__(self, n_plots, title=None, labels=None,
                 colors=None, line_styles=None):
        self.n_plots = n_plots
        self.title = title
        # Configure pyplot figure and axes objects
        self.fig, self.ax = self.__setup_plot(n_plots, labels, colors, line_styles)
        self.__configure_legend()

    def __setup_plot(self, n_plots, labels, colors, line_styles):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, N_EPOCHS)
        ax.set_ylim(0.0, 1.0)
        n_c = len(colors) if colors is not None else 0
        n_ls = len(line_styles) if line_styles is not None else 0
        n_labels = len(labels) if labels is not None else 0
        # Initialize lineplots
        for i in range(n_plots):
            ls = line_styles[i] if 0 <= i < n_ls else None
            c = colors[i] if 0 <= i < n_c else None
            label = labels[i] if 0 <= i < n_labels else None
            ax.plot([0], [0], linestyle=ls, color=c, label=label)
        ax.legend()
        ax.set_title(self.title)
        fig.canvas.draw_idle()
        return fig, ax

    def __configure_legend(self):
        pass

    def plot(self, i, data):
        new_x = np.arange(i+1)
        lines = self.ax.get_lines()
        for idx, new_y in enumerate(data):
            lines[idx].set_data(new_x, new_y)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)


def setup_logging(log_level=logging.INFO):
    log_handler = logging.FileHandler('log.txt', mode='w', encoding='utf-8')
    log_handler.setLevel(log_level)
    log_handler.setFormatter(DispatchingFormatter({
            'general': logging.Formatter('%(levelname)s: %(asctime)s - %(message)s'),
            'general.array': logging.Formatter('')
        },
        logging.Formatter('%(levelname)s: %(message)s')))
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(log_level)


def get_statistics(network_fit, input_fit, test_population_fit=None):
    result = ((network_fit.max(), network_fit.mean(), network_fit.std()),
              (input_fit.max(), input_fit.mean(), input_fit.std()))
    if test_population_fit is not None:
        result += ((test_population_fit.max(),
                    test_population_fit.mean(),
                    test_population_fit.std()),)
    return result

def collect_and_log_fitness(network_fit, input_fit,
                            test_population_fit=None, collector=None):
    log = logging.getLogger('general')
    array_log = logging.getLogger('general.array')
    stats = get_statistics(network_fit, input_fit,
                           test_population_fit=test_population_fit)
    net_max, net_mean, net_std = stats[0]
    input_max, input_mean, input_std = stats[1]
    log.debug('Network fitness:')
    array_log.debug(network_fit)
    log.info('Network best: %f', net_max)
    log.info('Mean network fitness: %f', net_mean)
    log.info('Std. dev. network fitness: %f', net_std)
    log.info('Input best: %f', input_max)
    log.info('Mean input fitness: %f', input_mean)
    log.info('Std. dev. input fitness: %f', input_std)
    if test_population_fit is not None:
        test_max, test_mean, test_std = stats[2]
        log.debug('Network fitness (on test input):')
        array_log.debug(test_population_fit)
        log.info('Network best (on test input): %f', test_max)
        log.info('Mean network fitness (on test input): %f', test_mean)
        log.info('Std. dev. network fitness (on test input): %f', test_std)
    if collector is not None:
        if isinstance(collector, dict):
            collector['net_max'].append(net_max)
            collector['net_mean'].append(net_mean)
            collector['net_std'].append(net_std)
            collector['input_max'].append(input_max)
            collector['input_mean'].append(input_mean)
            collector['input_std'].append(input_std)
            if test_population_fit is not None:
                collector['test_max'].append(test_max)
                collector['test_mean'].append(test_mean)
                collector['test_std'].append(test_std)
        if isinstance(collector, pd.DataFrame):
            row = np.array([s for stat in stats for s in stat])
            collector.loc[collector.shape[0]] = row
