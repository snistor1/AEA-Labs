import random
import time
from utils import *
from subsumption_checker import subsumes, generate_subsumption_graph_matrix
import warnings

warnings.filterwarnings("ignore")


def is_redundant(net, n):
    return outputs(net, n) == outputs(net[:-1], n)


def bad(net_outputs, n):
    bad = 0
    for out in net_outputs:
        # get no. of zeros
        out_len = len(out)
        p = 0
        for i in range(out_len):
            p += out[i] == 0
        for i in range(p):
            bad += out[i] == 1
        for i in range(p, out_len):
            bad += out[i] == 0
    return bad


def f(net, n):
    outs = outputs(net, n)
    return ((2 ** n) * bad(outs, n) + len(outs) - n - 1) / ((n + 1) * (2 ** n - 1))


def green_filter(n):
    G = []
    length = 1
    while length < n:
        for k in range(length):
            for i in range(k, n - length, 2 * length):
                G.append((i, i + length))
        length *= 2
    return tuple(G)


def filter_subsumed(R, C_best, n):
    return {Cp for Cp in R
            if not subsumes(C_best, n, Cp, n)}


def filter_lower(R, C_best, n, bound):
    new_R = set()
    new_R_sz = 0
    while R:
        Cp = R.pop()
        if new_R_sz + 1 < bound:
            new_R.add(Cp)
            new_R_sz += 1
        else:
            f_C_best = f(C_best, n)
            f_Cp = f(Cp, n)
            if f_C_best > f_Cp:
                x = random.random()
                if x < (f_C_best - f_Cp):
                    new_R.add(Cp)
                    new_R_sz += 1
    return new_R


def filter_all_subsumed(R, n):
    subsumed_set = set()
    for C, Cp in itertools.combinations(R, 2):
        if subsumes(C, n, Cp, n):
            subsumed_set.add(Cp)
    return {C for C in R if C not in subsumed_set}


def sortnet_best(n, k, bound, F=None, eps=1e-9):
    q = len(F) if hasattr(F, "__len__") else 0
    R = [None] * (k + 1)
    R[q] = {F} if F is not None else {tuple()}
    if F is not None:
        R[q] = {F}
        if f(F, n) < eps:
            return (R, F)
    else:
        R[q] = {tuple()}
    R[q] = {F} if F is not None else {tuple()}
    for p in range(q + 1, k + 1):
        R[p] = set()
        for C in R[p - 1]:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if is_redundant(C + ((i, j),), n):
                        continue
                    C_opt = C + ((i, j),)
                    # Remove all networks subsumed by C* from R_p^n...
                    R[p] = filter_subsumed(R[p], C_opt, n)
                    # ...and networks with a lower value.
                    R[p] = filter_lower(R[p], C_opt, n, bound)
                    R[p].add(C_opt)
        R[p] = filter_all_subsumed(R[p], n)
        sorted_R = sorted(R[p], key=lambda r: f(r, n))
        sorted_R = sorted_R[:bound]
        R[p] = set(sorted_R)
        for C in R[p]:
            if f(C, n) < eps:
                return (R, C)
    return (R, tuple())


if __name__ == '__main__':
    k = 25
    bound = 500
    for n in range(2, 9):
        start = time.time()
        history, sortnet = sortnet_best(n, k, bound, F=green_filter(n))
        elapsed_time = time.time() - start
        print('-'*20)
        print("n =", n)
        print(sortnet)
        print(len(sortnet))
        print("Elapsed time: %.6f" % (elapsed_time))
