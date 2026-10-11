"""One pass over all connected graphs of order n, cached on disk.

For every graph the pass records LE(G) and e_L(G), and along the way it
collects everything the paper states about exhaustive data:
  * the sign theorem (no negative local energy with d_v <= avgd/2),
  * the range corollary (LE(G-v) >= 0, equality iff G-v is edgeless),
  * the minimum local energy and its minimisers (Problem on the minimum),
  * the variational lemma and the majorisation lemma are checked separately
    (they need full spectra) in lle.checks.sign.
Results are written to results/cache/order_<n>[_part<k>of<m>].npz so that the
order-ten pass (several minutes) runs only once.
"""
import json
import os
import time

import numpy as np

from .scan import stream

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cache")


def _path(n, res, mod):
    tag = "" if mod == 1 else "_part%dof%d" % (res, mod)
    return os.path.join(CACHE, "order_%d%s" % (n, tag))


def collect(n, chunk=100000, parts=1, tol=1e-9, use_cache=True, only_part=None, progress=True):
    """Scan order n (optionally split into `parts` geng res/mod pieces) and return a dict."""
    os.makedirs(CACHE, exist_ok=True)
    pieces = []
    for res in range(parts):
        if only_part is not None and res != only_part:
            continue
        base = _path(n, res, parts)
        if use_cache and os.path.exists(base + ".npz") and os.path.exists(base + ".json"):
            z = np.load(base + ".npz")
            st = json.load(open(base + ".json"))
            pieces.append((z["g6"], z["LE"], z["EL"], st))
            continue
        t0 = time.time()
        g6s, LEs, ELs = [], [], []
        st = dict(graphs=0, pairs=0, negative=0, negative_light=0, shift_sum=0.0,
                  le_del_negative=0, eq_vs_edgeless=0, edgeless_vs_star=0,
                  min=float("inf"), minimizers={})
        resmod = None if parts == 1 else "%d/%d" % (res, parts)
        for bt, A, g6, le, ld, deg, m in stream(n, chunk=chunk, resmod=resmod, progress=progress):
            loc = le[:, None] - ld
            abar = (2.0 * m / n)[:, None]
            g6s.append(g6.copy()); LEs.append(le); ELs.append(loc.sum(axis=1))
            st["graphs"] += A.shape[0]; st["pairs"] += A.shape[0] * n
            neg = loc < -tol
            st["negative"] += int(neg.sum())
            st["negative_light"] += int((neg & (deg <= abar / 2 + 1e-12)).sum())
            st["shift_sum"] += float(((abar - 2 * deg) / max(n - 1, 1))[neg].sum())
            st["le_del_negative"] += int((ld < -tol).sum())
            edgeless = np.isclose(m[:, None] - deg, 0.0)
            st["eq_vs_edgeless"] += int(((ld < tol) != edgeless).sum())
            starc = np.isclose(deg, n - 1) & np.isclose(m, n - 1)[:, None]
            st["edgeless_vs_star"] += int((edgeless != starc).sum())
            c = float(loc.min())
            if c < st["min"] - tol:
                st["min"], st["minimizers"] = c, {}
            if c <= st["min"] + tol:
                gi, vi = np.nonzero(loc <= st["min"] + tol)
                for g, v in zip(gi[:20], vi[:20]):
                    st["minimizers"][g6[g].tobytes().decode()] = float(loc[g, v])
        st["secs"] = time.time() - t0
        G6 = np.concatenate(g6s) if g6s else np.zeros((0, 1), np.uint8)
        LE = np.concatenate(LEs) if LEs else np.zeros(0)
        EL = np.concatenate(ELs) if ELs else np.zeros(0)
        np.savez(base + ".npz", g6=G6, LE=LE, EL=EL)
        json.dump(st, open(base + ".json", "w"), indent=1)
        pieces.append((G6, LE, EL, st))
    # merge
    G6 = np.concatenate([p[0] for p in pieces])
    LE = np.concatenate([p[1] for p in pieces])
    EL = np.concatenate([p[2] for p in pieces])
    st = dict(graphs=0, pairs=0, negative=0, negative_light=0, shift_sum=0.0, le_del_negative=0,
              eq_vs_edgeless=0, edgeless_vs_star=0, min=float("inf"), minimizers={}, secs=0.0)
    for p in pieces:
        s = p[3]
        for k in ("graphs", "pairs", "negative", "negative_light", "shift_sum",
                  "le_del_negative", "eq_vs_edgeless", "edgeless_vs_star", "secs"):
            st[k] += s[k]
    best = min(p[3]["min"] for p in pieces) if pieces else float("inf")
    st["min"] = best
    for p in pieces:
        for g, val in p[3]["minimizers"].items():
            if val <= best + tol:
                st["minimizers"][g] = val
    st["mean_shift"] = st["shift_sum"] / st["negative"] if st["negative"] else None
    return dict(n=n, g6=G6, LE=LE, EL=EL, stats=st)
