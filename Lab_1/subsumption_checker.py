import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from networkx import bipartite


def plot_graph(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pos = [(ii[1], ii[0]) for ii in graph.nodes()]
    pos_dict = dict(zip(graph.nodes(), pos))
    nx.draw(graph, pos=pos_dict, ax=ax, with_labels=True)
    plt.show(block=False)


def enum_maximum_matching(g):
    s1 = set(n for n, d in g.nodes(data=True) if d['bipartite'] == 0)
    s2 = set(g) - s1
    n1 = len(s1)
    nodes = list(s1) + list(s2)

    adj = nx.adjacency_matrix(g, nodes).tolil()
    all_matches = []

    # ----------------Find one matching----------------
    match = bipartite.hopcroft_karp_matching(g)

    matchadj = np.zeros(adj.shape).astype('int')
    for kk, vv in match.items():
        matchadj[nodes.index(kk), nodes.index(vv)] = 1
    matchadj = sparse.lil_matrix(matchadj)

    all_matches.append(matchadj)

    # -----------------Enter recursion-----------------
    all_matches = enum_maximum_matching_iter(adj, matchadj, all_matches, n1, None, True)

    # ---------------Re-orient match arcs---------------
    all_matches2 = []
    for ii in all_matches:
        match_list = sparse.find(ii[:n1] == 1)
        m1 = [nodes[jj] for jj in match_list[0]]
        m2 = [nodes[jj] for jj in match_list[1]]
        match_list = zip(m1, m2)

        all_matches2.append(match_list)

    return all_matches2


def enum_maximum_matching_iter(adj, matchadj, all_matches, n1, add_e=None, check_cycle=True):
    # -------------------Find cycles-------------------
    if check_cycle:
        d = matchadj.multiply(adj)
        d[n1:, :] = adj[n1:, :] - matchadj[n1:, :].multiply(adj[n1:, :])

        dg = nx.from_numpy_matrix(d.toarray(), create_using=nx.DiGraph())
        cycles = list(nx.simple_cycles(dg))
        if len(cycles) == 0:
            check_cycle = False
        else:
            check_cycle = True

    # if len(cycles)>0:
    if check_cycle:
        cycle = cycles[0]
        cycle.append(cycle[0])
        cycle = zip(cycle[:-1], cycle[1:])

        # --------------Create a new matching--------------
        new_match = matchadj.copy()
        for ee in cycle:
            if matchadj[ee[0], ee[1]] == 1:
                new_match[ee[0], ee[1]] = 0
                new_match[ee[1], ee[0]] = 0
                e = ee
            else:
                new_match[ee[0], ee[1]] = 1
                new_match[ee[1], ee[0]] = 1

        if add_e is not None:
            for ii in add_e:
                new_match[ii[0], ii[1]] = 1

        all_matches.append(new_match)

        # -----------------Form subproblems-----------------
        g_plus = adj.copy()
        g_minus = adj.copy()
        g_plus[e[0], :] = 0
        g_plus[:, e[1]] = 0
        g_plus[:, e[0]] = 0
        g_plus[e[1], :] = 0
        g_minus[e[0], e[1]] = 0
        g_minus[e[1], e[0]] = 0

        add_e_new = [e, ]
        if add_e is not None:
            add_e_new.extend(add_e)

        all_matches = enum_maximum_matching_iter(g_minus, new_match, all_matches, n1, add_e, check_cycle)
        all_matches = enum_maximum_matching_iter(g_plus, matchadj, all_matches, n1, add_e_new, check_cycle)

    else:
        # ---------------Find uncovered nodes---------------
        uncovered = np.where(np.sum(matchadj, axis=1) == 0)[0]

        if len(uncovered) == 0:
            return all_matches

        # ---------------Find feasible paths---------------
        paths = []
        for ii in uncovered:
            aa = adj[ii, :].dot(matchadj)
            if aa.sum() == 0:
                continue
            paths.append((ii, int(sparse.find(aa == 1)[1][0])))
            if len(paths) > 0:
                break

        if len(paths) == 0:
            return all_matches

        # ----------------------Find e----------------------
        feas1, feas2 = paths[0]
        e = (feas1, int(sparse.find(matchadj[:, feas2] == 1)[0]))

        # ----------------Create a new match----------------
        new_match = matchadj.copy()
        new_match[feas2, :] = 0
        new_match[:, feas2] = 0
        new_match[feas1, e[1]] = 1
        new_match[e[1], feas1] = 1

        if add_e is not None:
            for ii in add_e:
                new_match[ii[0], ii[1]] = 1

        all_matches.append(new_match)

        # -----------------Form subproblems-----------------
        g_plus = adj.copy()
        g_minus = adj.copy()
        g_plus[e[0], :] = 0
        g_plus[:, e[1]] = 0
        g_plus[:, e[0]] = 0
        g_plus[e[1], :] = 0
        g_minus[e[0], e[1]] = 0
        g_minus[e[1], e[0]] = 0

        add_e_new = [e, ]
        if add_e is not None:
            add_e_new.extend(add_e)

        all_matches = enum_maximum_matching_iter(g_minus, matchadj, all_matches, n1, add_e, check_cycle)
        all_matches = enum_maximum_matching_iter(g_plus, new_match, all_matches, n1, add_e_new, check_cycle)

    return all_matches


def find_cycle(adj, n1):
    path = []
    visited = set()

    def visit(v):
        if v in visited:
            return False
        visited.add(v)
        path.append(v)
        neighbours = sparse.find(adj[v, :] == 1)[1]
        for nn in neighbours:
            if nn in path or visit(nn):
                return True
        path.remove(v)
        return False

    nodes = range(n1)
    result = any(visit(v) for v in nodes)
    return result, path


def generate_subsumption_graph_matrix():
    return list()


def check_subsumption(matchings: list, b_output: set):
    if len(matchings) == 0:
        return False
    # TODO: need to check if any matching is in the output of Cb and if so return false


def generate_output_set():
    return set()


def main_subsumption_checker():
    a_output, b_output = generate_output_set(), generate_output_set()
    # subsumption_edges = generate_subsumption_graph_matrix()
    g = nx.Graph()
    subsumption_edges = [
        [(1, 0), (0, 0)],
        [(1, 0), (0, 1)],
        [(1, 0), (0, 2)],
        [(1, 1), (0, 0)],
        [(1, 2), (0, 2)],
        [(1, 2), (0, 5)],
        [(1, 3), (0, 2)],
        [(1, 3), (0, 3)],
        [(1, 4), (0, 3)],
        [(1, 4), (0, 5)],
        [(1, 5), (0, 2)],
        [(1, 5), (0, 4)],
        [(1, 5), (0, 6)],
        [(1, 6), (0, 1)],
        [(1, 6), (0, 4)],
        [(1, 6), (0, 6)]
    ]
    for edge in subsumption_edges:
        g.add_node(edge[0], bipartite=0)
        g.add_node(edge[1], bipartite=1)
    g.add_edges_from(subsumption_edges)
    all_matches = enum_maximum_matching(g)
    check_subsumption(all_matches, b_output)


if __name__ == '__main__':
    main_subsumption_checker()
