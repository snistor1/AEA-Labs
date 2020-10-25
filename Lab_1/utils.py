import itertools


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