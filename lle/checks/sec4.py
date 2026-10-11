"""Section 4: closed forms for classical families, with their structural readings."""
from fractions import Fraction as F
import random

import networkx as nx
import numpy as np
import sympy as sp
from mpmath import mp, mpf, pi, cot, csc, floor, acos

from .. import exact, energy as en, families as fam, graphs as gr
from ..exhaustive import collect
from ..report import Report

TOL = 1e-9


def _srg_tests():
    return [("Petersen", nx.petersen_graph()), ("Clebsch", gr.clebsch()),
            ("3x3 rook (Paley 9)", gr.rook(3)), ("4x4 rook", gr.rook(4)),
            ("Shrikhande", gr.shrikhande()), ("Paley 13", gr.paley(13)), ("Paley 17", gr.paley(17))]


def run(cfg):
    R = Report("SECTION 4  Classical graph families")
    mp.dps = 50

    # ------------------------------------------------------------- 4.1 multipartite
    worst = 0.0
    for N in range(2, int(cfg["MULTIPARTITE_N_MAX"]) + 1):
        for parts in sp.utilities.iterables.partitions(N):
            ps = [k for k, c in parts.items() for _ in range(c)]
            if len(ps) < 2:
                continue
            worst = max(worst, abs(float(fam.Phi(ps)) - en.LE(gr.complete_multipartite(ps))))
    R.check("Proposition (complete multipartite): LE = Phi(n), all partitions of N <= %d" % cfg["MULTIPARTITE_N_MAX"],
            worst < TOL, "max deviation %.1e" % worst)
    ok = all(exact.LE_exact(gr.star(n)) == fam.star_LE(n)
             and exact.LE_exact(gr.star(n)) - exact.LE_exact(gr.star(n - 1)) == fam.star_leaf(n)
             and exact.combine(exact.e_L_terms(gr.star(n)))[0] == 2 * (2 * n - 3)
             for n in range(3, int(cfg["STAR_N_MAX"]) + 1))
    R.check("Corollary (stars): LE, leaf value and e_L, exact, n <= %d" % cfg["STAR_N_MAX"], ok)
    ok = all(exact.LE_exact(nx.complete_bipartite_graph(p, q)) == fam.Lam(p, q)
             and (p == q or exact.combine(exact.e_L_terms(nx.complete_bipartite_graph(p, q)))[0] == fam.eL_Kpq(p, q))
             for p in range(1, 7) for q in range(p, 9))
    R.check("Corollary (complete bipartite): Lambda(p,q) and e_L(K_{p,q}), exact, p <= 6, q <= 8", ok)
    ok = all(exact.LE_exact(gr.complete_multipartite([r] * t))
             - exact.LE_exact(gr.complete_multipartite([r] * (t - 1) + [r - 1] if r > 1 else [r] * (t - 1)))
             == fam.balanced_local(r, t) for r in range(1, 6) for t in range(2, 6))
    R.check("Corollary (balanced multipartite): local energy, exact, r <= 5, t <= 5", ok)

    # ------------------------------------------------------------- range corollary
    ex_, nu_ = int(cfg["KRR_EXACT_R_MAX"]), int(cfg["KRR_R_MAX"])
    ok = all(exact.LE_exact(nx.complete_bipartite_graph(r, r)) - exact.LE_exact(nx.complete_bipartite_graph(r - 1, r))
             == fam.Krr_local(2 * r) for r in range(1, ex_ + 1))
    worst = 0.0
    for r in range(ex_ + 1, nu_ + 1, int(cfg["KRR_NUM_STEP"])):
        val = en.LE(nx.complete_bipartite_graph(r, r)) - en.LE(nx.complete_bipartite_graph(r - 1, r))
        worst = max(worst, abs(val - float(fam.Krr_local(2 * r))) / r)
    R.check("Corollary (range): K_{r,r} value -n/2+5/2+1/(2(n-1)), exact r <= %d, numerical r <= %d" % (ex_, nu_),
            ok and worst < 1e-9)
    bad, mins = 0, []
    for n in range(2, int(cfg["MAX_ORDER"]) + 1):
        st = collect(n, parts=int(cfg["PARTS"]) if n == 10 else 1, progress=False)["stats"]
        bad += st["le_del_negative"] + st["eq_vs_edgeless"] + st["edgeless_vs_star"]
        mins.append((n, st["min"], list(st["minimizers"])))
    R.check("Corollary (range): equality LE_G(v) = LE(G) exactly at star centres, all graphs of order <= %d"
            % cfg["MAX_ORDER"], bad == 0)
    ok = True
    for n, mn, gs in mins:
        if n >= 4 and n % 2 == 0:
            isK = [nx.is_isomorphic(gr.from_g6(s), nx.complete_bipartite_graph(n // 2, n // 2)) for s in gs]
            good = abs(mn - float(fam.Krr_local(n))) < 1e-8 and any(isK)
            ok &= good
            R.note("order %2d : min LE_G(v) = %+.6f, attained by K_{%d,%d}: %s (%d minimiser(s))"
                   % (n, mn, n // 2, n // 2, any(isK), len(gs)))
        elif n >= 5:
            R.note("order %2d : min LE_G(v) = %+.6f  (odd order; minimiser %s)" % (n, mn, gs[0]))
    R.check("Problem (minimum): K_{n/2,n/2} attains the minimum for every even order 4..%d" % cfg["MAX_ORDER"], ok)

    # ------------------------------------------------------------- 4.2 strongly regular
    okS = okI = True
    for name, G in _srg_tests():
        n, k, lam, mu = gr.srg_parameters(G)
        v = 0
        Nv = list(G.neighbors(v)); V2 = [u for u in G if u != v and u not in Nv]
        A = nx.to_numpy_array(G, nodelist=list(G))
        AN = A[np.ix_(Nv, Nv)]; B = A[np.ix_(Nv, V2)]; A2 = A[np.ix_(V2, V2)]
        th, W = np.linalg.eigh(AN)
        ths = sorted(th)[::-1]
        pred = fam.srg_deleted_spectrum(n, k, lam, mu, ths[1:])
        true = en.laplacian_spectrum(gr.delete(G, v))
        okS &= len(pred) == len(true) and np.allclose([float(p) for p in pred], true, atol=1e-9)
        for t, w in zip(th, W.T):
            if abs(w.sum()) > 1e-8:
                continue
            beta = k - mu + (lam - mu) * t - t * t
            okI &= np.allclose(B @ (B.T @ w), beta * w, atol=1e-9) and beta > -1e-9
            okI &= np.allclose(A2 @ (B.T @ w), (lam - mu - t) * (B.T @ w), atol=1e-9)
        R.note("%-20s srg(%d,%d,%d,%d)" % (name, n, k, lam, mu))
    R.check("Theorem (SRG deletion spectrum): predicted = actual, 7 strongly regular graphs", okS)
    R.check("Theorem (SRG): BB^T w = beta_j w with beta_j = ||B^T w||^2/||w||^2 >= 0, "
            "and A_{V2} B^T w = (lambda - mu - theta_j) B^T w", okI)
    P = nx.petersen_graph(); C = gr.clebsch()
    lp, lc = en.LE_mp(gr.delete(P, 0)), en.LE_mp(gr.delete(C, 0))
    R.check("Example (Petersen, Clebsch): LE(P-v) = 20/3 + 4 sqrt 3, LE(C-v) = 28/3 + 4 sqrt 21",
            abs(lp - (mpf(20) / 3 + 4 * mp.sqrt(3))) < mpf(10) ** -40
            and abs(lc - (mpf(28) / 3 + 4 * mp.sqrt(21))) < mpf(10) ** -40,
            "local energies %s , %s" % (mp.nstr(16 - lp, 10), mp.nstr(30 - lc, 10)))
    ok = True
    for q in cfg["PALEY_QS"]:
        G = gr.paley(q); Nv = list(G.neighbors(0))
        th = sorted(np.linalg.eigvalsh(nx.to_numpy_array(G.subgraph(Nv))))[::-1][1:]
        val = en.LE_mp(G) - en.LE_mp(gr.delete(G, 0))
        f = fam.conference_local(q, th)
        ok &= abs(val - f) < 1e-9 and val >= fam.conference_bound(q) - 1e-12 and val > 0
        R.note("Paley %2d : LE_G(v) = %s  >=  bound %s" % (q, mp.nstr(val, 10), mp.nstr(fam.conference_bound(q), 10)))
    R.check("Corollary (conference graphs): formula, positivity and lower bound, Paley %s" % cfg["PALEY_QS"], ok)

    # ------------------------------------------------------------- 4.3 paths and cycles
    worst = max(abs(float(fam.Pi(l)) - en.LE(nx.path_graph(l))) for l in range(2, int(cfg["PATH_L_MAX"]) + 1))
    R.check("Proposition (paths): simplified Pi_l = LE(P_l), l <= %d" % cfg["PATH_L_MAX"], worst < TOL,
            "max deviation %.1e" % worst)
    mp.dps = 30
    bad = sum(1 for l in range(2, int(cfg["THRESHOLD_L_MAX"]) + 1)
              if int(floor(mpf(l) / pi * acos(mpf(1) / l))) != (l + 1) // 2 - 1)
    R.check("Proposition (paths): old threshold floor((l/pi) arccos(1/l)) = ceil(l/2) - 1, l <= %d"
            % cfg["THRESHOLD_L_MAX"], bad == 0)
    mp.dps = 50
    worst = max(abs(float(fam.Gam(n)) - en.LE(nx.cycle_graph(n))) for n in range(3, int(cfg["CYCLE_GRAPH_N_MAX"]) + 1))
    pos = all(en.local(nx.cycle_graph(n), 0) > 0 for n in range(3, int(cfg["CYCLE_GRAPH_N_MAX"]) + 1))
    R.check("Proposition (cycles): Gamma_n = LE(C_n), positivity on actual graphs, n <= %d"
            % cfg["CYCLE_GRAPH_N_MAX"], worst < TOL and pos)
    mp.dps = 30
    four = 4 / pi
    allpos, mn, dev = True, (mpf(10), 0), mpf(0)
    for n in range(3, int(cfg["CYCLE_CF_N_MAX"]) + 1):
        v = fam.cycle_local(n)
        allpos &= v > 0
        if v < mn[0]:
            mn = (v, n)
        if n >= 10:
            dev = max(dev, abs(n * (four - v)))
    R.check("Corollary (cycles): positive for 3 <= n <= %d, minimum 2/3 at n = 4, n|4/pi - LE| bounded"
            % cfg["CYCLE_CF_N_MAX"], allpos and mn[1] == 4 and abs(mn[0] - mpf(2) / 3) < 1e-20 and dev < 5,
            "max n|4/pi - LE_C(v)| = %s ; LE_{C_1000}(v) = %s" % (mp.nstr(dev, 8), mp.nstr(fam.cycle_local(1000), 10)))
    chain = all(fam.Gam(2 * p) - fam.Pi(2 * p - 1) > fam.phi(p) for p in range(2, int(cfg["PHI_P_MAX"]) + 1))
    incr = all(fam.phi(p + 1) > fam.phi(p) for p in range(2, int(cfg["PHI_P_MAX"]) + 1))
    gp = int(cfg["GRID_POINTS"])
    gcot = all(cot(a) > 1 / a - a / 2 for a in (pi / 4 * k / gp for k in range(1, gp + 1)))
    gcsc = all(csc(b) - 1 / b <= pi * b / 18 for b in (pi / 6 * k / gp for k in range(1, gp + 1)))
    R.check("Corollary (cycles): proof chain for even n, phi(2) = %s > 0, phi increasing" % mp.nstr(fam.phi(2), 10),
            fam.phi(2) > 0 and chain and incr and gcot and gcsc)

    # ------------------------------------------------------------- 4.4 corona
    mp.dps = 50
    rnd = random.Random(1)
    ok = True
    for _ in range(int(cfg["CORONA_RANDOM"])):
        n = rnd.randint(3, 8)
        G = nx.gnp_random_graph(n, 0.5, seed=rnd.randint(0, 10 ** 6))
        if not nx.is_connected(G):
            continue
        pred = fam.corona_spectrum(list(en.laplacian_spectrum(G)))
        ok &= np.allclose([float(p) for p in pred], en.laplacian_spectrum(gr.corona(G)), atol=1e-9)
    R.check("Proposition (corona spectrum): G o K_1 for %d random connected G" % cfg["CORONA_RANDOM"], ok)
    x = sp.Symbol("x")
    ok = all(exact.charpoly(gr.laplacian(gr.delete(gr.corona_Kn(n), n)))
             == sp.Poly(x * (x ** 2 - (n + 2) * x + n) ** (n - 2) * (x ** 2 - (n + 2) * x + 2 * n - 1), x).all_coeffs()
             for n in range(3, int(cfg["CORONA_CHARPOLY_N_MAX"]) + 1))
    R.check("Theorem (K_n o K_1): pendant-deletion charpoly x(x^2-(n+2)x+n)^(n-2)(x^2-(n+2)x+2n-1), n <= %d"
            % cfg["CORONA_CHARPOLY_N_MAX"], ok)
    worst, two = 0.0, True
    for n in range(3, int(cfg["CORONA_GRAPH_N_MAX"]) + 1):
        G = gr.corona_Kn(n)
        prof = en.profile(G)
        worst = max(worst, abs(en.LE(G) - float(fam.corona_LE(n))),
                    abs(en.local(G, 0) - float(fam.corona_core(n))), abs(en.local(G, n) - float(fam.corona_pendant(n))))
        two &= len(set(np.round(prof, 8))) == 2
    R.check("Theorem (K_n o K_1): LE, core and pendant values, two atoms, n <= %d" % cfg["CORONA_GRAPH_N_MAX"],
            worst < TOL and two, "max deviation %.1e" % worst)
    ok = all(abs(en.local(gr.corona_Kn(3), 0) - 3.182675) < 1e-6 and abs(en.local(gr.corona_Kn(3), 3) - 1.369483) < 1e-6
             and abs(en.local(gr.corona_Kn(4), 0) - 5.491020) < 1e-6 and abs(en.local(gr.corona_Kn(4), 4) - 2.072280) < 1e-6
             for _ in [0])
    R.check("Example (K_3 o K_1, K_4 o K_1): numerical profiles", ok)
    mx = mpf(0)
    for n in cfg["CORONA_ASYMP_NS"]:
        a, b = fam.corona_core(n), fam.corona_pendant(n); N = 2 * n; s = a - b
        terms = [n * (a - (mpf(3) * n / 2 - mpf(1) / 4)), n * (b - (mpf(n) / 2 + mpf(1) / 4)),
                 N * (s - mpf(N - 1) / 2), (s / 2) ** 2 - mpf(N - 1) ** 2 / 16]
        mx = max([mx] + [abs(t) for t in terms])
    R.check("Corollary (corona spread): all four error terms bounded (< 2) for n in %s" % cfg["CORONA_ASYMP_NS"], mx < 2)

    # ------------------------------------------------------------- 4.5 clique joins and windmills
    worst = 0.0
    for a in range(1, 4):
        for bs in ([2, 2], [2, 3], [1, 2, 3], [3, 3, 3], [2, 2, 2, 2], [4, 1]):
            worst = max(worst, abs(float(fam.Omega(a, bs)) - en.LE(gr.clique_join(a, bs))))
    R.check("Proposition (clique joins): LE = Omega(a; b)", worst < TOL)
    ok = True
    for b in range(2, int(cfg["WINDMILL_B_MAX"]) + 1):
        for r in range(1, int(cfg["WINDMILL_R_MAX"]) + 1):
            G = gr.windmill(r, b)
            ok &= exact.LE_exact(G) == fam.windmill_LE(r, b)
            ok &= exact.combine(exact.e_L_terms(G))[0] == fam.windmill_eL(r, b)
            neg = fam.windmill_eL(r, b) < 0
            cond = b >= 4 and r > F(2 * b * b - 4 * b + 1, b * (b - 3))
            ok &= neg == cond
            if r >= 2:
                ok &= exact.LE_exact(G) - exact.LE_exact(gr.delete(G, 0)) == fam.windmill_hub(r, b)
                ok &= exact.LE_exact(G) - exact.LE_exact(gr.delete(G, 1)) == fam.windmill_blade(r, b)
    R.check("Corollary (windmills): LE, e_L, negativity condition, hub and blade values, exact, b <= %d, r <= %d"
            % (cfg["WINDMILL_B_MAX"], cfg["WINDMILL_R_MAX"]), ok)
    return R.summary()
