"""Closed forms of Section 4 (and the star/K_{r,r} values), in mpmath."""
from fractions import Fraction

from mpmath import mp, mpf, sqrt, cot, csc, pi, fabs

mp.dps = 50


# ---------------------------------------------------------------- 4.1
def Phi(parts):
    """LE(K_{n_1,...,n_k})  (Proposition: complete multipartite graphs)."""
    N = sum(parts)
    k = len(parts)
    S2 = Fraction(sum(p * p for p in parts))
    return N + (k - 2) * S2 / N + sum((p - 1) * abs(S2 / N - p) for p in parts)


def Lam(p, q):
    if p == 0 or q == 0:
        return Fraction(0)
    return Fraction((p + q) ** 2 + abs(p - q) * (2 * p * q - p - q), p + q)


def eL_Kpq(p, q):
    return Fraction(2 * p * (2 * q - 1) * (q - p + 1), p + q - 1)


def star_LE(n):
    return Fraction(2 * (n * n - 2 * n + 2), n)


def star_leaf(n):
    return Fraction(2 * (n - 2) * (n + 1), n * (n - 1))


def balanced_local(r, t):
    return Fraction(-2 * (t - 1) * (r * r - 3 * r + 1), t * r - 1)


def Krr_local(n):                      # n = 2r
    return Fraction(-n, 2) + Fraction(5, 2) + Fraction(1, 2 * (n - 1))


# ---------------------------------------------------------------- 4.2  (strongly regular)
def srg_eigen(n, k, lam, mu):
    D = (lam - mu) ** 2 + 4 * (k - mu)
    sq = sqrt(mpf(D))
    r = ((lam - mu) + sq) / 2
    s = ((lam - mu) - sq) / 2
    f = ((n - 1) - (2 * k + (n - 1) * (lam - mu)) / sq) / 2
    g = ((n - 1) + (2 * k + (n - 1) * (lam - mu)) / sq) / 2
    return r, s, f, g


def srg_deleted_spectrum(n, k, lam, mu, thetas):
    """Theorem 'Deletion spectrum of a strongly regular graph' (thetas = theta_1..theta_{k-1})."""
    r, s, f, g = srg_eigen(n, k, lam, mu)
    out = [mpf(0), mpf(k - 1 - lam + mu)]
    out += [k - r] * int(round(f - k)) + [k - s] * int(round(g - k))
    for th in thetas:
        d = sqrt((1 + lam - mu) ** 2 + 4 * (k - lam + mpf(th)))
        c = mpf(2 * k - 1 - lam + mu)
        out += [(c + d) / 2, (c - d) / 2]
    return sorted(out)


def conference_local(n, thetas):
    return (n - 1) * sqrt(mpf(n)) / 2 - sum(sqrt(n + 3 + 4 * mpf(t)) for t in thetas)


def conference_bound(n):
    return mpf(n - 1) / 2 * (sqrt(mpf(n)) - sqrt(mpf(n - 3)))


# ---------------------------------------------------------------- 4.3  (paths and cycles)
def Pi(l):
    t = pi / (2 * l)
    return 2 * cot(t) if l % 2 == 0 else 2 * csc(t) - mpf(2) / l


def Gam(n):
    if n % 4 == 0:
        return 4 * cot(pi / n)
    if n % 4 == 2:
        return 4 * csc(pi / n)
    return 2 * csc(pi / (2 * n))


def cycle_local(n):
    return Gam(n) - Pi(n - 1)


def phi(p):
    return 4 / pi - pi / p + mpf(2) / (2 * p - 1) - pi ** 2 / 54


# ---------------------------------------------------------------- 4.4  (corona)
def corona_spectrum(lams):
    """Proposition: spectrum of G o K_1 from spec L(G) = lams (with lams[0] = 0)."""
    out = [mpf(0), mpf(2)]
    for l in lams[1:]:
        l = mpf(l)
        d = sqrt(l * l + 4)
        out += [(l + 2 + d) / 2, (l + 2 - d) / 2]
    return sorted(out)


def corona_core(n):
    n = mpf(n)
    return ((n - 1) * (1 + sqrt(n * n + 4))
            - (2 * n * (n - 1) / (2 * n - 1) + fabs(2 - n * (n - 1) / (2 * n - 1))
               + (n - 2) * sqrt((n - 1) ** 2 + 4)))


def corona_pendant(n):
    n = mpf(n)
    return (n * n - 4 * n + 3) / (2 * n - 1) + sqrt(n * n + 4) - sqrt(n * n - 4 * n + 8)


def corona_LE(n):
    return (n - 1) * (1 + sqrt(mpf(n * n + 4)))


# ---------------------------------------------------------------- 4.5  (clique joins, windmills)
def Omega(a, bs):
    B = sum(bs)
    N = a + B
    r = len(bs)
    d = Fraction(a * (a - 1) + sum(b * (b - 1) for b in bs) + 2 * a * B, N)
    return d + a * abs(N - d) + (r - 1) * abs(a - d) + sum((b - 1) * abs(a + b - d) for b in bs)


def windmill_LE(r, b):
    return Fraction(2 * (b * b * r * r + b * r - r + 1), b * r + 1)


def windmill_eL(r, b):
    return Fraction(2 * (b + 1)) if r == 1 else Fraction(2 * (-b * b * r + 2 * b * b + 3 * b * r - 4 * b + 1))


def windmill_hub(r, b):
    return Fraction(2 * (b * r * r + 1), b * r + 1)


def windmill_blade(r, b):
    return Fraction(2 * (-(b * b - 3 * b + 1) * r * r + (2 * b * b - 5 * b + 4) * r + 2 * (b - 2)),
                    r * (b * r + 1))
