import multiprocessing as mproc
import logging

import numpy as np


global_mp_vars = {}


def eval_input(network, input_test_case) -> np.float64:
    result = input_test_case.copy()
    for comp in network:
        if input_test_case[comp[0]] > input_test_case[comp[1]]:
            result[[comp[0], comp[1]]] = result[[comp[1], comp[0]]]
    return np.all(result[:-1] <= result[1:]).astype(np.float64)


def init_worker(mat, mat_shape):
    global_mp_vars['mat'] = mat
    global_mp_vars['mat_shape'] = mat_shape


def worker(first, last, net_pop, input_pop):
    tmp = np.frombuffer(global_mp_vars['mat'], dtype=np.float64) \
            .reshape(global_mp_vars['mat_shape'])
    for i, net in enumerate(net_pop):
        for j, input_case in enumerate(input_pop):
            val = eval_input(net, input_case)
            tmp[first+i, j] = val


def evaluate(population: list, input_population: list,
             multiprocessing: bool = False) -> np.ndarray:
    net_pop_size = len(population)
    input_pop_size = len(input_population)
    if multiprocessing:
        ctype = np.ctypeslib.as_ctypes_type(np.float64)
        shared_matrix = mproc.RawArray(ctype, net_pop_size * input_pop_size)
        fit_matrix = np.frombuffer(shared_matrix, np.float64) \
                       .reshape((net_pop_size, input_pop_size))
        n_procs = mproc.cpu_count()
        step = np.ceil(net_pop_size / n_procs).astype(int)
        initargs = (shared_matrix, (net_pop_size, input_pop_size))
        with mproc.Pool(processes=n_procs, initializer=init_worker,
                        initargs=initargs) as pool:
            for i in range(n_procs):
                first = step * i
                last = step * (i + 1)
                args = (first, last,
                        population[first:last],
                        input_population)
                pool.apply_async(worker, args=args)
            pool.close()
            pool.join()
        net_fit, input_fit = (np.sum(fit_matrix, axis=1) / input_pop_size,
                              1 - np.sum(fit_matrix, axis=0) / net_pop_size)
        return net_fit, input_fit
    else:
        # int? shouldn't it be np.float64?
        fit_matrix = np.empty((net_pop_size, input_pop_size), dtype=int)
        for i, net in enumerate(population):
            for j, input_case in enumerate(input_population):
                fit_matrix[i, j] = eval_input(net, input_case)
        net_fit, input_fit = (np.sum(fit_matrix, axis=1) / input_pop_size,
                              1 - np.sum(fit_matrix, axis=0) / net_pop_size)
        return net_fit, input_fit
