import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from networkx import bipartite
from utils import outputs


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
    match = bipartite.hopcroft_karp_matching(g, top_nodes=s1)

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


def clusters(c_output, n):
    out_clusters = {i: set() for i in range(n + 1)}
    for out in c_output:
        out_clusters[sum(out)].add(out)
    return out_clusters


def zeros(c_out, n):
    return [tuple(map(min, zip(*v)))
            for _, v in clusters(c_out, n).items()]


def ones(c_out, n):
    return [tuple(map(max, zip(*v)))
            for _, v in clusters(c_out, n).items()]


def generate_subsumption_graph_matrix(ca_output: tuple, na: int,
                                      cb_output: tuple, nb: int):
    edges = []
    ca_zeros = zeros(ca_output, na)
    cb_zeros = zeros(cb_output, nb)
    ca_ones = ones(ca_output, na)
    cb_ones = ones(cb_output, nb)
    nclusters = len(ca_zeros)
    for i in range(na):
        for j in range(nb):
            valid = True
            for p in range(nclusters):
                if (cb_zeros[p][j] - ca_zeros[p][i] == 1
                        or ca_ones[p][i] - cb_ones[p][j] == 1):
                    valid = False
                    break
            if valid:
                edges.append((i + 1, -(j + 1)))
    return edges


def check_subsumption(matchings: list, a_output: set, b_output: set):
    # TODO:
    # Let Ca and Cb be n channel comparator networks. If there exists 1 ≤ k ≤ n
    # such that the number of sequences with k 1s in outputs(Ca) is greater than
    # that in outputs(Cb), then Ca !< Cb. (!< is "not subsumed")
    #
    # Experiments show that, in the context of this paper, more than 70% of the
    # subsumption tests in the application of the Prune algorithm are eliminated
    # based on [the aforementioned lemma].
    if len(matchings) == 0:
        return False
    for i in range(len(matchings)):
        # construct pi(outputs(Ca))
        perm_idxs = [-pi_x for _, pi_x in sorted(matchings[i])]
        pi_a_output = {tuple(elem[pi_x - 1] for pi_x in perm_idxs)
                       for elem in a_output}
        # check if subsets
        if pi_a_output.issubset(b_output):
            return True
    return False


def subsumes(ca: tuple, na: int, cb: tuple, nb: int):
    ca_output, cb_output = outputs(ca, na), outputs(cb, nb)
    subsumption_edges = generate_subsumption_graph_matrix(ca_output, na, cb_output, nb)
    g = nx.Graph()
    g.add_nodes_from(range(1, na + 1), bipartite=0)
    g.add_nodes_from(range(-1, -(nb + 1), -1), bipartite=1)
    g.add_edges_from(subsumption_edges)
    all_matches = enum_maximum_matching(g)
    return check_subsumption(all_matches, ca_output, cb_output)


if __name__ == '__main__':
    # ca = ((0,1),(2,3),(1,3),(1,4))
    # cb = ((0,1),(2,3),(0,3),(1,4))
    ca = ((0, 1), (1, 2), (0, 3))
    cb = ((0, 1), (0, 2), (1, 3))
    # This is supposed to be true
    print(subsumes(ca, 4, cb, 4))
