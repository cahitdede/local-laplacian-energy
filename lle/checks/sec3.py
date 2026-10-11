"""Section 3: the sign of the local energy."""
from fractions import Fraction as F

import networkx as nx
import numpy as np

from .. import exact
from ..exhaustive import collect
from ..report import Report
from ..scan import stream


def run(cfg):
    R = Report("SECTION 3  The sign of the local energy")
    # Lemma (variational form) and Lemma (deletion majorisation): full spectra needed
    nmax = int(cfg["LEMMA_N_MAX"])
    bad_var = bad_maj = total = 0
    for n in range(2, nmax + 1):
        for bt, A, g6, le, ld, deg, m in stream(n, progress=False):
            L = -A.copy(); L[:, bt.diag, bt.diag] = deg
            ev = np.sort(np.linalg.eigvalsh(L), axis=1)[:, ::-1]           # decreasing
            S = np.concatenate([np.zeros((len(ev), 1)), np.cumsum(ev, axis=1)], axis=1)
            abar = (2 * m / n)[:, None]
            var = 2 * (S - np.arange(n + 1)[None, :] * abar).max(axis=1)
            bad_var += int((np.abs(var - le) > 1e-9).sum())
            for v in range(n):
                Av = A[:, bt.keep[v]][:, :, bt.keep[v]]
                Lv = -Av; Lv[:, bt.diag1, bt.diag1] = Av.sum(axis=2)
                evv = np.sort(np.linalg.eigvalsh(Lv), axis=1)[:, ::-1]
                Sv = np.cumsum(evv, axis=1)
                bad_maj += int((Sv > S[:, 1:n] + 1e-9).sum())
            total += len(le)
    R.check("Lemma (variational form): LE = 2 max_k (S_k - k avgd), all graphs of order <= %d" % nmax,
            bad_var == 0, "%d graphs, %d violations" % (total, bad_var))
    R.check("Lemma (deletion majorisation): S_k(G-v) <= S_k(G), all graphs of order <= %d" % nmax,
            bad_maj == 0, "%d violations" % bad_maj)
    # Theorem (necessary condition) on exhaustive data
    rows, bad, tot_pairs, tot_neg = [], 0, 0, 0
    for n in range(2, int(cfg["MAX_ORDER"]) + 1):
        st = collect(n, parts=int(cfg["PARTS"]) if n == 10 else 1, progress=False)["stats"]
        bad += st["negative_light"]; tot_pairs += st["pairs"]; tot_neg += st["negative"]
        rows.append((n, st["graphs"], st["negative"], st["mean_shift"]))
    R.check("Theorem (necessary condition): no vertex with d_v <= avgd/2 has negative local energy",
            bad == 0, "%d (G,v) pairs of order <= %d checked ; %d negative instances, all with d_v > avgd/2"
            % (tot_pairs, cfg["MAX_ORDER"], tot_neg))
    for n, g, neg, sh in rows:
        if neg:
            R.note("order %2d : %9d graphs, %7d vertices with negative local energy, mean baseline shift %+.4f"
                   % (n, g, neg, sh))
    # sharpness witness K_n
    ok = all(exact.LE_exact(nx.complete_graph(n)) - exact.LE_exact(nx.complete_graph(n - 1)) == 2
             for n in range(2, int(cfg["KN_N_MAX"]) + 1))
    R.check("Sharpness: LE_{K_n}(v) = 2 for 2 <= n <= %d (exact)" % cfg["KN_N_MAX"], ok)
    return R.summary()
