"""The finite-order census of Section 5 (table of ~_L-classes) and Remark on
logical independence, reproduced from the cached exhaustive scans.

Classes are formed by sorting on LE and cutting where consecutive values differ
by more than `tol`, then cutting again on e_L.  A gap analysis compares the
largest spread inside accepted classes with the smallest rejected gap.
Laplacian cospectrality is decided exactly through the integer power sums
tr L^k; profile equality is certified exactly with lle.exact.
"""
import csv
import json
import os

import numpy as np

from . import exact
from .graphs import from_g6
from .exhaustive import collect, CACHE
from .scan import Batch

RESULTS = os.path.dirname(CACHE)


def classes(LE, EL, tol):
    order = np.argsort(LE, kind="stable")
    les = LE[order]
    cut = np.nonzero(np.diff(les) > tol)[0] + 1
    out = []
    for g in np.split(order, cut):
        if len(g) < 2:
            continue
        o = g[np.argsort(EL[g], kind="stable")]
        cc = np.nonzero(np.diff(EL[o]) > tol)[0] + 1
        out += [s for s in np.split(o, cc) if len(s) > 1]
    gaps = np.diff(les)
    rejected = gaps[(gaps > tol) & (gaps < 1e-3)]
    spread = max((max(LE[c].max() - LE[c].min(), EL[c].max() - EL[c].min()) for c in out), default=0.0)
    return out, float(spread), (float(rejected.min()) if rejected.size else float("nan"))


def _invariants(n, g6rows):
    bt = Batch(n)
    buf = b"".join(r.tobytes() + b"\n" for r in g6rows)
    A, _ = bt.decode(buf)
    le, ld, deg, m = bt.energies(A)
    prof = np.sort(le[:, None] - ld, axis=1)
    L = np.rint(-A).astype(np.int64)
    L[:, bt.diag, bt.diag] = np.rint(deg).astype(np.int64)
    assert n <= 12, "int64 trace keys are safe only up to order 12"
    P = np.zeros_like(L); P[:, bt.diag, bt.diag] = 1
    keys = np.empty((L.shape[0], n), dtype=np.int64)
    for t in range(n):
        P = np.matmul(P, L)
        keys[:, t] = np.trace(P, axis1=1, axis2=2)
    return np.rint(m).astype(int), np.sort(np.rint(deg).astype(int), axis=1), keys, prof


def order_row(n, tol=1e-9, parts=1, certify=True, write=True):
    d = collect(n, parts=parts, progress=False)
    cls, spread, gap = classes(d["LE"], d["EL"], tol)
    sizes = np.array([len(c) for c in cls], dtype=int)
    row = dict(order=n, classes=len(cls), graphs=int(sizes.sum()) if len(cls) else 0,
               pairs=int((sizes * (sizes - 1) // 2).sum()), max_class=int(sizes.max()) if len(cls) else 0,
               max_spread=spread, min_rejected_gap=gap)
    if not cls:
        row.update(equal_m=0, equal_deg=0, cospectral=0, equal_profile=0, cospectral_and_profile=0)
        return row, []
    idx = np.concatenate(cls)
    m, ds, keys, prof = _invariants(n, d["g6"][idx])
    pos = {int(g): k for k, g in enumerate(idx)}
    eq_m = eq_d = cos = eq_p = both = 0
    pairs = []
    for c in cls:
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                i, j = pos[int(c[a])], pos[int(c[b])]
                eq_m += m[i] == m[j]
                eq_d += np.array_equal(ds[i], ds[j])
                cs = np.array_equal(keys[i], keys[j]); cos += cs
                ps = bool(np.all(np.abs(prof[i] - prof[j]) <= 1e-9)); eq_p += ps
                both += cs and ps
                if ps:
                    pairs.append((d["g6"][c[a]].tobytes().decode(), d["g6"][c[b]].tobytes().decode(), bool(cs)))
    row.update(equal_m=int(eq_m), equal_deg=int(eq_d), cospectral=int(cos),
               equal_profile=int(eq_p), cospectral_and_profile=int(both))
    # largest class
    big = max(cls, key=len)
    bi = [pos[int(g)] for g in big]
    row["largest_class"] = dict(size=len(big),
                                sizes=sorted(set(int(m[i]) for i in bi)),
                                distinct_degree_sequences=len({tuple(ds[i]) for i in bi}),
                                distinct_charpolys=len({tuple(keys[i]) for i in bi}),
                                distinct_profiles=len({tuple(np.round(prof[i], 8)) for i in bi}))
    cert = []
    if certify:
        for a, b, cs in pairs:
            Ga, Gb = from_g6(a), from_g6(b)
            full = exact.same_profile(Ga, Gb)
            noref = exact.same_profile(Ga, Gb, reflect=False)
            rat = exact.profile_exact(Ga) is not None and exact.profile_exact(Gb) is not None
            prof_ex = exact.profile_exact(Ga) if rat else None
            cert.append(dict(g1=a, g2=b, cospectral=cs, certified=bool(full), rational=rat,
                             needs_reflection=bool(full) and not noref,
                             profile=[str(x) for x in prof_ex] if prof_ex else None))
        row["certified"] = sum(c["certified"] for c in cert)
        row["rational_profiles"] = sum(c["rational"] for c in cert)
        row["needs_reflection"] = sum(c["needs_reflection"] for c in cert)
    if write:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "classes_order_%d.txt" % n), "w") as f:
            f.write("# class_id graph6   (all ~_L classes of connected graphs of order %d)\n" % n)
            for k, c in enumerate(cls):
                for g in c:
                    f.write("%d %s\n" % (k, d["g6"][g].tobytes().decode()))
        with open(os.path.join(RESULTS, "profile_pairs_order_%d.csv" % n), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["g1", "g2", "cospectral", "certified_exactly", "rational_profile",
                        "needs_reflection", "exact_profile"])
            for c in cert:
                w.writerow([c["g1"], c["g2"], c["cospectral"], c["certified"], c["rational"],
                            c["needs_reflection"], " ".join(c["profile"]) if c["profile"] else ""])
    return row, cert


def independence_counts(n, tol=1e-9):
    """(#pairs with equal e_L and different LE, #pairs with equal LE and different e_L), exactly certified."""
    d = collect(n, progress=False)
    LE, EL, g6 = d["LE"], d["EL"], d["g6"]
    N = len(LE)
    a = b = 0
    unsettled = []
    for i in range(N):
        for j in range(i + 1, N):
            eqE, eqL = abs(EL[i] - EL[j]) < tol, abs(LE[i] - LE[j]) < tol
            if eqE == eqL:
                continue
            si, sj = g6[i].tobytes().decode(), g6[j].tobytes().decode()
            Gi, Gj = from_g6(si), from_g6(sj)
            if eqE:
                ok = exact.same_value(exact.e_L_terms(Gi), exact.e_L_terms(Gj))
                a += 1
            else:
                ok = exact.same_value([(1, exact.laplacian(Gi))], [(1, exact.laplacian(Gj))])
                b += 1
            if ok is None:
                unsettled.append(("e_L" if eqE else "LE", si, sj))
    return a, b, unsettled
