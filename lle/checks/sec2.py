"""Section 2: logical independence of LE and e_L, and the examples."""
from fractions import Fraction as F

import networkx as nx
import numpy as np

from .. import exact, energy as en
from ..census import independence_counts
from ..exhaustive import collect
from ..graphs import from_g6, laplacian
from ..report import Report
from .common import connected_graphs, agree_mp, eL_graph_terms

PAPER_COUNTS = {3: (1, 0), 4: (0, 3), 5: (10, 4), 6: (15, 152)}


def run(cfg):
    R = Report("SECTION 2  Deletion-energy profiles")
    P3, K3 = nx.path_graph(3), nx.complete_graph(3)
    paw = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    diamond = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    eLx = lambda G: exact.combine(exact.e_L_terms(G))[0]
    R.check("Remark (logical independence): P_3, K_3 values",
            exact.LE_exact(P3) == F(10, 3) and exact.LE_exact(K3) == 4 and eLx(P3) == 6 and eLx(K3) == 6,
            "LE(P3)=%s  LE(K3)=%s  e_L=%s, %s" % (exact.LE_exact(P3), exact.LE_exact(K3), eLx(P3), eLx(K3)))
    R.check("Remark (logical independence): paw and diamond",
            exact.LE_exact(paw) == 6 and exact.LE_exact(diamond) == 6
            and eLx(paw) == F(32, 3) and eLx(diamond) == F(28, 3),
            "e_L(paw)=%s  e_L(diamond)=%s" % (eLx(paw), eLx(diamond)))
    for n in range(3, int(cfg["IND_N_MAX"]) + 1):
        a, b, uns = independence_counts(n)
        exp = PAPER_COUNTS.get(n)
        worst = 0.0
        for kind, s1, s2 in uns:
            G1, G2 = from_g6(s1), from_g6(s2)
            ta = eL_graph_terms(G1) if kind == "e_L" else [(1, G1)]
            tb = eL_graph_terms(G2) if kind == "e_L" else [(1, G2)]
            worst = max(worst, float(agree_mp(ta, tb)))
        R.check("Remark (logical independence): counts at order %d" % n,
                (exp is None or (a, b) == exp) and worst < 1e-40,
                "equal e_L & different LE: %d ; equal LE & different e_L: %d ; paper %s ; "
                "%d equalities not settled by symbols agree to 60 digits (max diff %.1e)"
                % (a, b, exp, len(uns), worst))
    # (a)
    A, B = from_g6("EEvw"), from_g6("EE~w")
    pa, pb = exact.profile_exact(A), exact.profile_exact(B)
    R.check("Example (a): EEvw, EE~w",
            exact.LE_exact(A) == 12 == exact.LE_exact(B)
            and exact.combine(exact.e_L_terms(A))[0] == F(102, 5) == exact.combine(exact.e_L_terms(B))[0]
            and pa == sorted([F(14, 5)] * 2 + [F(18, 5)] * 3 + [F(4)])
            and pb == sorted([F(16, 5)] * 3 + [F(18, 5)] * 3),
            "profiles %s | %s" % ([str(x) for x in pa], [str(x) for x in pb]))
    # (b)
    C, D = from_g6("FCQVw"), from_g6("FCQvw")
    pc = exact.profile_exact(C)
    R.check("Example (b): FCQVw, FCQvw",
            exact.LE_exact(C) == F(92, 7) == exact.LE_exact(D) and exact.same_profile(C, D)
            and pc == sorted([F(8, 7)] * 2 + [F(22, 7)] * 4 + [F(36, 7)])
            and int(np.trace(laplacian(C))) == 20 and int(np.trace(laplacian(D))) == 22,
            "profile %s ; traces 20, 22" % [str(x) for x in pc])
    # central moment duality on all graphs of small order
    worst = 0.0
    for n in range(3, int(cfg["DUALITY_N_MAX"]) + 1):
        for G in connected_graphs(n):
            prof = np.array(en.profile(G)); dk = np.array(en.deck(G))
            for k in range(1, 5):
                lhs = ((prof - prof.mean()) ** k).sum()
                rhs = (-1) ** k * ((dk - dk.mean()) ** k).sum()
                worst = max(worst, abs(lhs - rhs))
    R.check("Theorem (central moments): duality for k=1..4, all graphs of order <= %d" % cfg["DUALITY_N_MAX"],
            worst < 1e-8, "max deviation %.1e" % worst)
    return R.summary()
