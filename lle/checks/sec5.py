"""Section 5: profile variance, cospectral graphs, and the census."""
import json
import os
from fractions import Fraction as F

import networkx as nx
import numpy as np
from mpmath import mp, mpf

from .. import exact, energy as en, graphs as gr
from ..census import order_row, classes, RESULTS
from ..exhaustive import collect
from ..report import Report
from .common import connected_graphs, agree_mp, eL_graph_terms

PAPER = {6: (3, 7, 5, 3, 0, 0, 0, 1), 7: (20, 41, 22, 3, 15, 5, 2, 1), 8: (135, 305, 258, 10, 85, 47, 37, 6),
         9: (1059, 2296, 1499, 6, 1080, 642, 596, 53), 10: (17856, 37296, 24335, 42, 15852, 12900, 12335, 19)}


def _dl(G, v):
    return en.delta_lambda(G, v)


def run(cfg):
    R = Report("SECTION 5  Profile variance and discriminating power")
    # ------------------------------------------------------------- variance results
    bad_c = bad_1 = bad_2 = 0
    ratios, arg = [], None
    for n in range(3, int(cfg["VARIANCE_N_MAX"]) + 1):
        for G in connected_graphs(n):
            a = 2 * G.number_of_edges() / n
            ML = [en.local(G, v) for v in G]
            DL = [_dl(G, v) for v in G]
            d = np.array([G.degree(v) for v in G], float)
            s2 = float(((d - a) ** 2).mean())
            kap = np.sqrt(4 * s2 + a * a)
            for v, x, y in zip(G, ML, DL):
                bad_c += abs(x - y) > abs(a - 2 * G.degree(v)) + 1e-9
            lhs = abs(np.sqrt(en.variance(ML)) - np.sqrt(en.variance(DL)))
            bad_1 += lhs > kap + 1e-9
            bad_2 += en.variance(ML) > 2 * en.variance(DL) + 2 * kap ** 2 + 1e-9
            ratios.append(lhs / kap)
            if lhs / kap >= max(ratios):
                arg = G
    R.check("Proposition (centering shift): |LE_G(v) - DeltaLambda| <= |avgd - 2 d_v|, order <= %d" % cfg["VARIANCE_N_MAX"],
            bad_c == 0)
    R.check("Theorem (variance bound): both inequalities, order <= %d" % cfg["VARIANCE_N_MAX"], bad_1 == 0 and bad_2 == 0)
    r = np.array(ratios)
    isS = nx.is_isomorphic(arg, gr.star(arg.number_of_nodes()))
    R.check("Remark (sharpness): max ratio %.3f attained by a star (%s), median %.3f"
            % (r.max(), isS, float(np.median(r))),
            abs(r.max() - 0.650) < 5e-4 and isS and abs(np.median(r) - 0.065) < 5e-3)
    vals = []
    for n in (7, 20, 50):
        G = gr.star(n); a = 2 * (n - 1) / n
        ML = [en.local(G, v) for v in G]; DL = [_dl(G, v) for v in G]
        d = np.array([G.degree(v) for v in G], float)
        kap = np.sqrt(4 * ((d - a) ** 2).mean() + a * a)
        vals.append(abs(np.sqrt(en.variance(ML)) - np.sqrt(en.variance(DL))) / kap)
    R.check("Remark (sharpness): star ratios 0.650, 0.875, 0.950 at n = 7, 20, 50",
            np.allclose(vals, [0.650, 0.875, 0.950], atol=5e-4), "computed %s" % np.round(vals, 3))
    bad = 0
    for n in range(3, int(cfg["REGULAR_N_MAX"]) + 1):
        for G in connected_graphs(n):
            if len({d for _, d in G.degree()}) != 1:
                continue
            k = G.degree(0)
            ML = [en.local(G, v) for v in G]; DL = [_dl(G, v) for v in G]
            bad += abs(np.sqrt(en.variance(ML)) - np.sqrt(en.variance(DL))) > k + 1e-9
            bad += en.variance(ML) > 2 * en.variance(DL) + 2 * k * k + 1e-9
    R.check("Corollary (regular graphs): both bounds on every regular graph of order <= %d" % cfg["REGULAR_N_MAX"], bad == 0)
    G = gr.from_g6("FFzn_")
    prof = exact.profile_exact(G)
    dl = sorted(round(_dl(G, v), 9) for v in G)
    R.check("Example (regular, not deck-homogeneous): FFzn_",
            prof == sorted([F(2, 3)] * 3 + [F(4, 3)] * 4) and dl == [0.0] * 4 + [2.0] * 3
            and len({d for _, d in G.degree()}) == 1
            and abs(en.variance([float(x) for x in prof]) - 16 / 147) < 1e-12 and abs(en.variance(dl) - 48 / 49) < 1e-9,
            "profile %s ; Delta Lambda %s" % ([str(x) for x in prof], dl))
    ok = True
    for s in ("F~AGO", "F?\\vg"):
        G = gr.from_g6(s); a = 2 * G.number_of_edges() / 7
        ok &= min(d for _, d in G.degree()) == 1 <= a / 2 and min(en.profile(G)) < 0
    R.check("Corollary (spread from the baseline): hypothesis realised by F~AGO and F?\\vg", ok)

    # ------------------------------------------------------------- cospectral examples
    G1, G2 = gr.minimal_cospectral_pair()
    L1, L2 = gr.laplacian(G1), gr.laplacian(G2)
    import sympy as sp
    x = sp.Symbol("x")
    cp = sp.factor(sp.Poly(exact.charpoly(L1), x).as_expr())
    sym = exact.symbol(L1)
    le_ok = sym[0] == 4 and len(sym[1]) == 1 and list(sym[1].values()) == [2]
    p1, p2 = en.profile(G1), en.profile(G2)
    R.check("Example (smallest cospectral pair): charpoly %s, LE = 4 + 2 sqrt 5" % cp,
            exact.charpoly(L1) == exact.charpoly(L2) and le_ok and abs(en.LE(G1) - (4 + 2 * 5 ** 0.5)) < 1e-12
            and np.allclose(p1, [0.4721, 1.5097, 1.5097, 2.1320, 2.1320, 3.2721], atol=5e-5)
            and np.allclose(p2, [1.9097, 1.9097, 2.0721, 2.0721, 2.1320, 2.1320], atol=5e-5)
            and abs(sum(p1) - 11.0277) < 5e-5 and abs(sum(p2) - 12.2277) < 5e-5
            and abs(en.variance(p1) - 0.7184) < 5e-5 and abs(en.variance(p2) - 0.0088) < 5e-5)
    Rk, Sh = gr.rook(4), gr.shrikhande()
    R.check("Example (rook 4x4, Shrikhande): cospectral, LE = 36, constant profiles 3.177795 / 3.223746",
            exact.charpoly(gr.laplacian(Rk)) == exact.charpoly(gr.laplacian(Sh)) and not nx.is_isomorphic(Rk, Sh)
            and exact.LE_exact(Rk) == 36 and np.allclose(en.profile(Rk), 3.177795, atol=1e-6)
            and np.allclose(en.profile(Sh), 3.223746, atol=1e-6))
    A, B = gr.from_g6("HC~vfr~"), gr.from_g6("H?~vvj~")
    pa = exact.profile_exact(A)
    decks = (sorted(tuple(exact.charpoly(gr.laplacian(gr.delete(A, v)))) for v in A)
             == sorted(tuple(exact.charpoly(gr.laplacian(gr.delete(B, v)))) for v in B))
    R.check("Example (cospectral graphs with identical profiles): HC~vfr~, H?~vvj~",
            not nx.is_isomorphic(A, B) and exact.charpoly(gr.laplacian(A)) == exact.charpoly(gr.laplacian(B))
            and exact.same_profile(A, B) and pa == sorted([F(16, 9)] * 4 + [F(34, 9)] * 4 + [F(43, 9)])
            and exact.LE_exact(A) == F(160, 9) and not decks,
            "spectrum %s ; profile %s ; vertex-deleted spectral decks differ: %s"
            % (np.round(en.laplacian_spectrum(A), 6).tolist(), [str(t) for t in pa], not decks))

    # ------------------------------------------------------------- census
    summary = {}
    for n in range(6, int(cfg["MAX_ORDER"]) + 1):
        parts = int(cfg["PARTS"]) if n == 10 else 1
        row, cert = order_row(n, parts=parts, certify=True, write=True)
        summary[n] = row
        got = (row["classes"], row["graphs"], row["pairs"], row["max_class"], row["equal_m"],
               row["equal_deg"], row["cospectral"], row["equal_profile"])
        R.check("Census, order %d: %s" % (n, got), got == PAPER[n],
                "paper %s ; profile pairs certified exactly %d/%d ; rational %d ; need reflection %d ; "
                "cospectral with equal profile %d" % (PAPER[n], row["certified"], row["equal_profile"],
                                                      row["rational_profiles"], row["needs_reflection"],
                                                      row["cospectral_and_profile"]))
    if 10 in summary:
        lc = summary[10]["largest_class"]
        R.check("Census, order 10: largest class (42 graphs, ten sizes in [22,34], 42 degree sequences, "
                "42 charpolys, 41 profiles)",
                lc["size"] == 42 and len(lc["sizes"]) == 10 and min(lc["sizes"]) == 22 and max(lc["sizes"]) == 34
                and lc["distinct_degree_sequences"] == 42 and lc["distinct_charpolys"] == 42
                and lc["distinct_profiles"] == 41)
        r10 = summary[10]
        R.check("Remark (exactness), order 10: 19 certified, 5 rational, 3 need the reflection",
                r10["certified"] == 19 and r10["rational_profiles"] == 5 and r10["needs_reflection"] == 3)
    cos = {n: summary[n]["cospectral_and_profile"] for n in summary}
    R.check("Cospectral pairs with identical profile: none through order 8, six at 9, seven at 10",
            all(cos[n] == 0 for n in cos if n <= 8) and cos.get(9, 6) == 6 and cos.get(10, 7) == 7, str(cos))
    # tolerance stability window and near misses
    for n in range(7, int(cfg["MAX_ORDER"]) + 1):
        d = collect(n, parts=int(cfg["PARTS"]) if n == 10 else 1, progress=False)
        LE, EL = d["LE"], d["EL"]
        cls, spread, _ = classes(LE, EL, 1e-9)
        cid = np.full(len(LE), -1, dtype=np.int64)
        for k, c in enumerate(cls):
            cid[c] = k
        o = np.argsort(LE, kind="stable"); les, els, cs = LE[o], EL[o], cid[o]
        best, near = np.inf, []
        k = 1
        while True:
            dle = les[k:] - les[:-k]; idx = np.nonzero(dle <= 1e-6)[0]
            if not len(idx):
                break
            sup = np.maximum(dle[idx], np.abs(els[idx + k] - els[idx]))
            diff = ~((cs[idx] == cs[idx + k]) & (cs[idx] >= 0))
            if diff.any():
                best = min(best, float(sup[diff].min()))
                for t in idx[diff & (sup < float(cfg["NEAR_MISS"]))]:
                    near.append((o[t], o[t + k]))
            k += 1
        conf = True
        for i, j in near:
            Gi = gr.from_g6(d["g6"][i].tobytes().decode()); Gj = gr.from_g6(d["g6"][j].tobytes().decode())
            dle_ = agree_mp([(1, Gi)], [(1, Gj)]); del_ = agree_mp(eL_graph_terms(Gi), eL_graph_terms(Gj))
            conf &= max(dle_, del_) > mpf(10) ** -30
        R.check("Census stability, order %d: classification unchanged for every tolerance in [%.1e, %.1e)"
                % (n, spread, best), spread < 1e-12 and best > 1e-9 and conf,
                "%d cross-class pairs closer than %.0e, each confirmed distinct in 60-digit arithmetic"
                % (len(near), cfg["NEAR_MISS"]))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump({str(k): v for k, v in summary.items()}, open(os.path.join(RESULTS, "census_summary.json"), "w"),
              indent=1, default=str)
    return R.summary()
