import networkx as nx
from mpmath import mp, fabs

from ..scan import stream
from ..graphs import from_g6, delete
from ..energy import LE_mp


def connected_graphs(n, chunk=100000):
    """networkx graphs for all connected graphs of order n (use only for small n)."""
    for bt, A, g6, *_ in stream(n, chunk=chunk, progress=False):
        for row in g6:
            yield from_g6(row.tobytes().decode())


def agree_mp(terms_a, terms_b, dps=60):
    """|sum a_i LE(F_i) - sum b_j LE(H_j)| evaluated with `dps` digits."""
    mp.dps = dps
    va = sum(c * LE_mp(F, dps) for c, F in terms_a)
    vb = sum(c * LE_mp(F, dps) for c, F in terms_b)
    mp.dps = dps
    return fabs(va - vb)


def eL_graph_terms(G):
    n = G.number_of_nodes()
    return [(n, G)] + [(-1, delete(G, v)) for v in G]
