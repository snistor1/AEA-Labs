import random
import itertools

def compare_swap(lst, i, j):
    if lst[i] > lst[j]:
        lst[i], lst[j] = lst[j], lst[i]

def sort(net, lst):
    for i, j in net:
        compare_swap(lst, i, j)

def outputs(net, n):
    sortees = [list(elem) for elem in itertools.product([0,1], repeat=n)]
    for sortee in sortees:
        sort(net, sortee)
    return set(map(tuple, sortees))

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
    return ((2**n) * bad(outs, n) + len(outs) - n - 1) / ((n+1)*(2**n-1))

def subsumes(net1, net2):
    return False

def green_filter(n):
    G = []
    length = 1
    while length < n:
        for k in range(1, length+1):
            for i in range(k, n-length+1, 2*length):
                G.append((i, i+length))
        length *= 2
    return tuple(G)

def sortnet_best(n, k, bound, F=None):
    q = len(F)
    R = [None] * k
    R[q] = {F}
    print(R)
    for p in range(q+1, k):
        R[p] = set()
        for C in R[p-1]:
            for i in range(1, n):
                for j in range(i+1, n):
                    if is_redundant(C + (i,j), n):
                        continue
                    C_opt = C + (i,j)
                    for Cp in R[p]:
                        if subsumes(C_opt, Cp):
                            R[p].remove(Cp)
                        if len(R[p]) >= bound and f(C_opt) < f(Cp):
                            x = random.random()
                            if f(C_opt) < x and f(Cp) > x:
                                R[p].remove(Cp)
                    R[p].add(C_opt)
        for C, Cp in itertools.combinations(R[p]):
            if subsumes(C, Cp):
                R[p].remove(Cp)
        sorted_R = sorted(R[p], key=lambda r: f(r))
        sorted_R = sorted_R[:len(sorted_R) - bound]
        R[p] = set(sorted_R)
    return R[k]


if __name__ == '__main__':
    # print(green_filter(10))
    # print(len(green_filter(10)))
    # sortnet_best(10, 29, 2000, F=green_filter(10))

    # my_example = [1,2,0]
    # print(my_example)
    # sort(((0,1),(0,2),(1,2)), my_example)
    # print(my_example)
    #get_outputs(((0,4), (1,3), (2,3), (4,1)), 5)
    print(f(((0,1), (2,3), (1,3), (0,2), (1,3)), 4))
