import random
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


def sortnet_best(n, k, bound, F=None):
    q = len(F) if hasattr(F, "__len__") else 0
    R = [None] * (k + 1)
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
                    R[p] = {Cp for Cp in R[p]
                            if not subsumes(C_opt, n, Cp, n)}
                    # ...and networks with a lower value.
                    new_R = set()
                    new_R_sz = 0
                    while R[p]:
                        Cp = R[p].pop()
                        if new_R_sz + 1 < bound:
                            new_R.add(Cp)
                            new_R_sz += 1
                        else:
                            f_C_opt = f(C_opt, n)
                            f_Cp = f(Cp, n)
                            if f_C_opt > f_Cp:
                                x = random.random()
                                if x < (f_C_opt - f_Cp):
                                    new_R.add(Cp)
                                    new_R_sz += 1
                    R[p] = new_R
                    R[p].add(C_opt)
        for C, Cp in itertools.combinations(R[p], 2):
            if subsumes(C, n, Cp, n):
                R[p].remove(Cp)
        sorted_R = sorted(R[p], key=lambda r: f(r, n))
        sorted_R = sorted_R[:bound]
        R[p] = set(sorted_R)
    return R


if __name__ == '__main__':
    n = 4
    k = 5
    bound = 200
    # print(sortnet_best(n, k, bound, F=green_filter(n)))
    found_sortnets = sortnet_best(n, k, bound)
    print(found_sortnets)
    print(found_sortnets[k])
