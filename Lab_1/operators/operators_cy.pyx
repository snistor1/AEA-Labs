import copy
from enum import IntEnum

import numpy as np
cimport numpy as np
cimport cython

from ga_defines import *


class MutationCases(IntEnum):
    REPLACE = 0
    INSERT = 1
    DELETE = 2

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void insert_comparators(np.int64_t[:, :, :] pop,
                             np.int64_t[:] indices,
                             np.int64_t[:] pos,
                             np.int64_t[:, ::1] comparators,
                             np.int64_t[:] lengths):
    cdef Py_ssize_t i
    cdef np.int64_t j
    cdef np.int64_t curr_pos
    cdef np.int64_t curr_len
    cdef np.int64_t curr_idx
    cdef Py_ssize_t pop_size = indices.size
    for i in range(pop_size):
        curr_len = lengths[i]
        if curr_len == MAX_COMPARATORS:
            continue
        curr_pos = pos[i]
        curr_idx = indices[i]
        for j in range(curr_len, curr_pos, -1):
            pop[curr_idx, j] = pop[curr_idx, j-1]
        pop[curr_idx, curr_pos, :] = comparators[i, :]


# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void delete_comparators(np.int64_t[:, :, :] pop,
                             np.int64_t[:] indices,
                             np.int64_t[:] pos,
                             np.int64_t[:] lengths):
    cdef Py_ssize_t i
    cdef np.int64_t j
    cdef np.int64_t curr_pos
    cdef np.int64_t curr_len
    cdef np.int64_t curr_idx
    cdef Py_ssize_t pop_size = indices.size
    for i in range(pop_size):
        curr_len = lengths[i]
        if curr_len == MIN_COMPARATORS:
            continue
        curr_pos = pos[i]
        curr_idx = indices[i]
        for j in range(curr_pos, curr_len):
            pop[curr_idx, j] = pop[curr_idx, j+1]


def fast_mutation_indices(n_genes, pm, K=2.0):
    rng = np.random.default_rng()
    idxs = np.cumsum(
        np.ceil(
            np.log(
                rng.random(np.ceil(K * pm * n_genes).astype(int))
            )
            /
            np.log(1 - pm)
        ).astype(int)
    )
    return idxs[:np.searchsorted(idxs, n_genes)]


def upgrade_net(net_population):
    crossover_net(net_population)
    mutate_net(net_population)


def crossover_net(population):
    rng = np.random.default_rng()
    n_networks = population.shape[0]
    selected_idxs = (rng.random(n_networks) <
                     NET_CROSSOVER_P).nonzero()[0]
    rng.shuffle(selected_idxs)
    if selected_idxs.size & 1 != 0:
        selected_idxs = selected_idxs[:-1]
    selected_pop = population[selected_idxs]
    selected_net_lengths = np.argmin(selected_pop, axis=1)[:, 0]
    cuts = rng.integers(np.min(selected_net_lengths.reshape(-1, 2), axis=1))
    for i in range(selected_idxs.size // 2):
        cut = cuts[i]
        tmp = selected_pop[2*i, :cut, :].copy()
        (selected_pop[2*i, :cut, :],
         selected_pop[2*i + 1, :cut, :]) = (selected_pop[2*i + 1, :cut, :],
                                            tmp)
    population[selected_idxs] = selected_pop


def mutate_net(population):
    rng = np.random.default_rng()
    n_networks = population.shape[0]
    # Normalize case probabilities
    cases_cumsum = np.cumsum(np.array(NET_MUTATION_CASES) / np.sum(NET_MUTATION_CASES))
    if NET_MUTATION_P < NET_FAST_MUTATION_THRESH:
        selected_idxs = fast_mutation_indices(n_networks, NET_MUTATION_P)
    else:
        selected_idxs = (rng.random(size=n_networks) <
                         NET_MUTATION_P).nonzero()[0]
    case_idxs = np.searchsorted(cases_cumsum, rng.random(selected_idxs.size))
    r_mask = case_idxs == MutationCases.REPLACE
    i_mask = case_idxs == MutationCases.INSERT
    d_mask = ~(r_mask | i_mask)
    n_rs, n_is = (np.count_nonzero(r_mask),
                  np.count_nonzero(i_mask))
    selected_net_lengths = np.argmin(population[selected_idxs], axis=1)[:, 0]
    new_comparators = np.sort(
        np.argpartition(
            rng.random((n_rs + n_is, NETWORK_SIZE)),
            2,
            axis=-1
        )[:, :2]
    )
    pos = rng.integers(selected_net_lengths)
    # Replace operation
    population[selected_idxs[r_mask], pos[r_mask]] = new_comparators[:n_rs]
    # Insert operation
    insert_comparators(population,
                       selected_idxs[i_mask],
                       pos[i_mask],
                       new_comparators[n_rs:],
                       selected_net_lengths[i_mask])
    # Delete operation
    delete_comparators(population,
                       selected_idxs[d_mask],
                       pos[d_mask],
                       selected_net_lengths[d_mask])


def upgrade_input(input_population):
    crossover_input(input_population)
    mutate_input(input_population)


def crossover_input(population):
    rng = np.random.default_rng()
    n_inputs = population.shape[0]
    selected_idxs = (rng.random(n_inputs) <
                     INPUT_CROSSOVER_P).nonzero()[0]
    rng.shuffle(selected_idxs)
    if selected_idxs.size & 1 != 0:
        selected_idxs = selected_idxs[:-1]
    selected_pop = population[selected_idxs]
    cuts = rng.integers(population.shape[1], size=selected_idxs.size // 2)
    for i in range(selected_idxs.size // 2):
        cut = cuts[i]
        tmp = selected_pop[2*i, :cut].copy()
        (selected_pop[2*i, :cut],
         selected_pop[2*i + 1, :cut]) = (selected_pop[2*i + 1, :cut],
                                         tmp)
    population[selected_idxs] = selected_pop


def mutate_input(population):
    rng = np.random.default_rng()
    if INPUT_MUTATION_P < INPUT_FAST_MUTATION_THRESH:
        selected_idxs = fast_mutation_indices(population.size,
                                              INPUT_MUTATION_P)
    else:
        selected_idxs = (rng.random(population.size) <
                         INPUT_MUTATION_P).nonzero()[0]
    population.ravel()[selected_idxs] ^= 1
