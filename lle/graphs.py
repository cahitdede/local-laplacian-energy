"""Graph constructors and conversions used throughout the paper."""
import itertools

import networkx as nx
import numpy as np


def from_g6(s):
    return nx.from_graph6_bytes(s.encode() if isinstance(s, str) else s)


def to_g6(G):
    H = nx.convert_node_labels_to_integers(G)
    return nx.to_graph6_bytes(H, header=False).decode().strip()


def laplacian(G):
    """Integer Laplacian matrix (numpy int64) in a fixed vertex order."""
    A = nx.to_numpy_array(G, nodelist=list(G), dtype=np.int64)
    return np.diag(A.sum(axis=1)) - A


def delete(G, v):
    H = G.copy()
    H.remove_node(v)
    return H


# ------------------------------------------------------------------ families
def star(n):                       # S_n = K_{1,n-1}, centre 0
    return nx.star_graph(n - 1)


def complete_multipartite(parts):
    return nx.complete_multipartite_graph(*parts)


def corona(G):
    """G o K_1: every vertex v receives a pendant vertex ('p', v)."""
    H = nx.Graph(G)
    for v in list(G):
        H.add_edge(v, ("p", v))
    return H


def corona_Kn(n):                  # core vertices 0..n-1, pendant of i is n+i
    G = nx.complete_graph(n)
    for i in range(n):
        G.add_edge(i, n + i)
    return G


def clique_join(a, bs):
    """J(a; b_1..b_r) = K_a v (K_{b_1} + ... + K_{b_r})."""
    G = nx.complete_graph(a)
    nxt = a
    for b in bs:
        blade = list(range(nxt, nxt + b))
        nxt += b
        G.add_edges_from(itertools.combinations(blade, 2))
        G.add_edges_from((c, x) for c in range(a) for x in blade)
        G.add_nodes_from(blade)
    return G


def windmill(r, b):                # W_{r,b} = K_1 v r K_b, hub 0
    return clique_join(1, [b] * r)


def paley(q):
    squares = {(x * x) % q for x in range(1, q)}
    G = nx.Graph()
    G.add_nodes_from(range(q))
    G.add_edges_from((a, c) for a in range(q) for c in range(a + 1, q) if (a - c) % q in squares)
    return G


def rook(m):                       # m x m rook's graph = line graph of K_{m,m}
    return nx.convert_node_labels_to_integers(nx.line_graph(nx.complete_bipartite_graph(m, m)))


def shrikhande():
    G = nx.Graph()
    gens = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    for a in range(4):
        for b in range(4):
            for da, db in gens:
                G.add_edge((a, b), ((a + da) % 4, (b + db) % 4))
    return nx.convert_node_labels_to_integers(G)


def clebsch():                     # folded 5-cube, srg(16,5,0,2)
    C = nx.hypercube_graph(5)
    anti = lambda t: tuple(1 - z for z in t)
    rep = {t: min(t, anti(t)) for t in C}
    G = nx.Graph()
    for u, w in C.edges():
        G.add_edge(rep[u], rep[w])
        G.add_edge(rep[u], rep[anti(w)])
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G)


def minimal_cospectral_pair():
    """Triangle and 4-cycle sharing a vertex; K_{2,3} plus a pendant at a degree-2 vertex."""
    G1 = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 0)])
    G2 = nx.complete_bipartite_graph(2, 3)
    G2.add_edge(2, 5)
    return G1, G2


def srg_parameters(G):
    """(n, k, lambda, mu) if G is strongly regular, else None."""
    n = G.number_of_nodes()
    degs = {d for _, d in G.degree()}
    if len(degs) != 1:
        return None
    k = degs.pop()
    A = nx.to_numpy_array(G, nodelist=list(G))
    A2 = A @ A
    lam = {int(round(A2[i, j])) for i in range(n) for j in range(n) if i != j and A[i, j]}
    mu = {int(round(A2[i, j])) for i in range(n) for j in range(n) if i != j and not A[i, j]}
    if len(lam) != 1 or len(mu) > 1:
        return None
    return n, k, lam.pop(), (mu.pop() if mu else 0)
