"""Exact arithmetic: characteristic polynomials, symbols of Laplacian energies,
and certificates for equalities between (integer combinations of) energies.

The *symbol* of a graph F is a pair (q, I) where q is rational and I is a
multiset of canonical pairs (f, t), f an irreducible factor of the Laplacian
characteristic polynomial normalised modulo rational translation and the
reflection x -> -x, and t the number of its roots above the baseline.  Then

    LE(F) = q + 2 * sum_{(f,t) in I} Sigma_{f,t},

where Sigma_{f,t} is the sum of the t largest roots of f.  Equal symbols imply
equal energies, and the test needs only integer factorisation, Sturm root
counting and rational arithmetic (paper, Remark "The coincidences are exact").
Comparing algebraic numbers by interval refinement cannot prove equality,
which is why this route is used.
"""
from collections import Counter
from fractions import Fraction

import numpy as np
import sympy as sp

from .graphs import laplacian, delete

__all__ = ['charpoly', 'trace_key', 'symbol', 'combine', 'LE_exact', 'profile_symbols',
           'profile_exact', 'same_profile', 'same_value', 'e_L_terms', 'laplacian']

X = sp.Symbol("x")


def charpoly(L):
    """Exact integer coefficients [1, c1, ..., cn] of det(xI - L) (Newton identities, Python ints)."""
    L = np.asarray(L, dtype=object)
    n = L.shape[0]
    P = np.identity(n, dtype=object)
    p = []
    for _ in range(n):
        P = P.dot(L)
        p.append(int(sum(P[i, i] for i in range(n))))
    e = [1]
    for k in range(1, n + 1):
        acc = sum((-1) ** (j - 1) * e[k - j] * p[j - 1] for j in range(1, k + 1))
        assert acc % k == 0
        e.append(acc // k)
    return [(-1) ** k * e[k] for k in range(n + 1)]


def trace_key(L):
    """(tr L, ..., tr L^n): equal keys  <=>  equal characteristic polynomials."""
    L = np.asarray(L, dtype=object)
    n = L.shape[0]
    P = np.identity(n, dtype=object)
    out = []
    for _ in range(n):
        P = P.dot(L)
        out.append(int(sum(P[i, i] for i in range(n))))
    return tuple(out)


def _fr(r):
    r = sp.Rational(r)
    return Fraction(int(sp.numer(r)), int(sp.denom(r)))


def _canon(pf, t, reflect=True):
    d = pf.degree()
    cs = pf.all_coeffs()
    mu = sp.Rational(-cs[1], cs[0]) / d
    f0 = sp.Poly(sp.expand(pf.as_expr().subs(X, X + mu)), X).monic()
    fm = sp.Poly(sp.expand((-1) ** d * f0.as_expr().subs(X, -X)), X).monic()
    c0 = tuple(_fr(c) for c in f0.all_coeffs())
    cm = tuple(_fr(c) for c in fm.all_coeffs())
    if not reflect:
        return (c0, t), _fr(mu)
    return ((c0, t) if c0 <= cm else (cm, d - t)), _fr(mu)


def symbol(L, reflect=True):
    """Symbol (q, Counter) of LE for an integer Laplacian matrix L."""
    L = np.asarray(L)
    n = L.shape[0]
    if n == 0:
        return Fraction(0), Counter()
    ab = Fraction(int(np.trace(L)), n)
    a = sp.Rational(ab.numerator, ab.denominator)
    _, facs = sp.factor_list(sp.Poly(charpoly(L), X).as_expr(), X)
    q, kstar, irr = Fraction(0), 0, Counter()
    for f, mult in facs:
        pf = sp.Poly(f, X)
        d = pf.degree()
        if d == 0:
            continue
        cs = pf.all_coeffs()
        rsum = sp.Rational(-cs[1], cs[0])
        t = pf.count_roots(inf=a) if d > 1 else (1 if rsum > a else 0)
        if t == 0:
            continue
        if t == d:
            q += mult * _fr(rsum)
            kstar += mult * d
        else:
            key, mu = _canon(pf, t, reflect)
            q += mult * t * mu
            kstar += mult * t
            irr[key] += mult
    return 2 * (q - kstar * ab), Counter({k: 2 * c for k, c in irr.items()})


def combine(terms, reflect=True):
    """Symbol of sum c_i * LE(F_i) for terms = [(c_i, L_i)]."""
    q, irr = Fraction(0), Counter()
    for c, L in terms:
        qi, ii = symbol(L, reflect)
        q += c * qi
        for k, m in ii.items():
            irr[k] += c * m
    return q, Counter({k: m for k, m in irr.items() if m != 0})


def LE_exact(G):
    """Exact LE as a Fraction when it is rational, else None."""
    q, irr = symbol(laplacian(G))
    return q if not irr else None


def profile_symbols(G, reflect=True):
    """[(symbol of LE_G(v))] for all v, each as (q, frozenset of items)."""
    qG, iG = symbol(laplacian(G), reflect)
    out = []
    for v in G:
        qv, iv = symbol(laplacian(delete(G, v)), reflect)
        irr = Counter(iG)
        for k, c in iv.items():
            irr[k] -= c
        out.append((qG - qv, frozenset((k, c) for k, c in irr.items() if c != 0)))
    return sorted(out, key=lambda s: (s[0], sorted(map(repr, s[1]))))


def profile_exact(G):
    """Exact profile as sorted Fractions when every local energy is rational, else None."""
    syms = profile_symbols(G)
    if any(s[1] for s in syms):
        return None
    return sorted(s[0] for s in syms)


def same_profile(G, H, reflect=True):
    """Certify M_L(G) = M_L(H) exactly (True), or return None if the symbols do not settle it."""
    return True if profile_symbols(G, reflect) == profile_symbols(H, reflect) else None


def same_value(terms_a, terms_b):
    """Certify sum a_i LE(F_i) == sum b_j LE(H_j); True if certified, None otherwise."""
    qa, ia = combine(terms_a)
    qb, ib = combine(terms_b)
    return True if (qa == qb and ia == ib) else None


def e_L_terms(G):
    """e_L(G) = n LE(G) - sum_v LE(G - v) as a list of (coefficient, Laplacian)."""
    n = G.number_of_nodes()
    return [(n, laplacian(G))] + [(-1, laplacian(delete(G, v))) for v in G]
