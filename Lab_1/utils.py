import itertools
import logging
from collections import defaultdict

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
    def __init__(self, n_plots, title=None, xlabel=None,
                 ylabel=None, labels=None, colors=None,
                 line_styles=None):
        self.n_plots = n_plots
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        # Configure pyplot figure and axes objects
        self.fig, self.ax = self.__setup_plot(n_plots, labels, colors, line_styles)

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
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid()
        fig.canvas.draw_idle()
        return fig, ax

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


# def get_statistics(network_fit, input_fit, test_population_fit=None):
#     result = ((network_fit.mean(), network_fit.std()),
#               (input_fit.mean(), input_fit.std()))
#     if test_population_fit is not None:
#         result += ((test_population_fit.mean(),
#                     test_population_fit.std()),)
#     return result

def collect(network_fit, input_fit,
            test_fit=None, global_abs=None,
            global_rel=None, collector=None):
    log = logging.getLogger('general')
    array_log = logging.getLogger('general.array')
    results_dict = defaultdict(lambda: 0.0)

    results_dict['net_mean'] = network_fit.mean()
    results_dict['net_std'] = network_fit.std()
    results_dict['input_mean']  = input_fit.mean()
    results_dict['input_std'] = input_fit.std()
    log.debug('Network fitness:')
    array_log.debug(network_fit)
    log.info('Mean network fitness: %f', results_dict['net_mean'])
    log.info('Std. dev. network fitness: %f', results_dict['net_std'])
    log.info('Mean input fitness: %f', results_dict['input_mean'])
    log.info('Std. dev. input fitness: %f', results_dict['input_std'])
    if test_fit is not None:
        results_dict['test_mean'] = test_fit.mean()
        results_dict['test_std'] = test_fit.std()
        log.debug('Network fitness (on test input):')
        array_log.debug(test_fit)
        log.info('Mean network fitness (on test input): %f', results_dict['test_mean'])
        log.info('Std. dev. network fitness (on test input): %f', results_dict['test_std'])
    if global_rel is not None:
        results_dict['global_max_rel'] = global_rel
        log.info('Global maximum (relative): %f', results_dict['global_max_rel'])
    if global_abs is not None:
        results_dict['global_max_abs'] = global_abs
        log.info('Global maximum (aboslute): %f', results_dict['global_max_abs'])
    return results_dict
