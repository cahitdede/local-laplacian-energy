"""Laplacian energy and local energies: double precision and arbitrary precision."""
import numpy as np
from mpmath import mp, mpf, matrix as mpmatrix, eigsy, fabs

from .graphs import laplacian, delete


def LE_matrix(L):
    """Laplacian energy of an integer/float Laplacian matrix (double precision)."""
    n = L.shape[0]
    if n == 0:
        return 0.0
    ev = np.linalg.eigvalsh(np.asarray(L, dtype=float))
    return float(np.abs(ev - np.trace(L) / n).sum())


def LE(G):
    return LE_matrix(laplacian(G))


def LE_mp(G, dps=50):
    """Laplacian energy to ``dps`` decimal digits (mpmath symmetric eigensolver)."""
    L = laplacian(G)
    n = L.shape[0]
    if n == 0:
        return mpf(0)
    mp.dps = int(dps)
    M = mpmatrix(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = mpf(int(L[i, j]))
    ev = eigsy(M, eigvals_only=True)
    a = mpf(int(np.trace(L))) / n
    return sum(fabs(ev[i] - a) for i in range(n))


def local(G, v, f=LE):
    return f(G) - f(delete(G, v))


def profile(G, f=LE):
    le = f(G)
    return sorted(le - f(delete(G, v)) for v in G)


def total_local(G, f=LE):
    return sum(profile(G, f))


def deck(G, f=LE):
    return sorted(f(delete(G, v)) for v in G)


def delta_lambda(G, v):
    """Uncentred deletion variation  LE(G) - sum_j |gamma_j - avgd(G)|."""
    n = G.number_of_nodes()
    a = 2.0 * G.number_of_edges() / n
    ev = np.linalg.eigvalsh(laplacian(delete(G, v)).astype(float))
    return LE(G) - float(np.abs(ev - a).sum())


def variance(xs):
    xs = np.asarray(xs, dtype=float)
    return float(np.mean((xs - xs.mean()) ** 2))


def laplacian_spectrum(G):
    return np.sort(np.linalg.eigvalsh(laplacian(G).astype(float)))
