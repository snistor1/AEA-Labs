import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


cdef extern from "<alloca.h>":
    void* alloca(size_t) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double eval_input_cython(np.int64_t[:, :] net, np.int64_t[:] inp) nogil:
    cdef size_t n_comparators = net.shape[0]
    cdef size_t n_input_entries = inp.shape[0]
    cdef np.int64_t c0, c1
    cdef np.int64_t tmp
    cdef Py_ssize_t i
    cdef np.int64_t* res = <np.int64_t*>alloca(n_input_entries * sizeof(np.int64_t))
    # copy data to buffer
    for i in range(n_input_entries):
        res[i] = inp[i]
    for i in range(n_comparators):
        c0 = net[i, 0]
        c1 = net[i, 1]
        if c0 == -1 or c1 == -1:
            break
        if res[c0] > res[c1]:
            tmp = res[c0]
            res[c0] = res[c1]
            res[c1] = tmp
    for i in range(n_input_entries - 1):
        if res[i] > res[i+1]:
            return 0.0
    return 1.0

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef double[:, ::1] evaluate_cython(np.int64_t[:, :, ::1] net_population,
                                    np.int64_t[:, ::1] input_population):
    cdef size_t net_pop_size = net_population.shape[0]
    cdef size_t input_pop_size = input_population.shape[0]
    cdef size_t input_case_size = input_population.shape[1]
    cdef np.ndarray fit_matrix = np.empty((net_pop_size, input_pop_size))
    cdef double[:, ::1] fit_view = fit_matrix
    cdef Py_ssize_t i, j
    for i in prange(net_pop_size, nogil=True):
        for j in range(input_pop_size):
            fit_view[i, j] = eval_input_cython(net_population[i],
                                               input_population[j])
    return fit_view

def evaluate(net_population, input_population):
    """
    Wrapper for cdef function.
    """
    net_pop_size = net_population.shape[0]
    input_pop_size = input_population.shape[0]
    result = np.asarray(evaluate_cython(net_population, input_population))
    return (np.sum(result, axis=1) / input_pop_size,
            1 - np.sum(result, axis=0) / net_pop_size)
