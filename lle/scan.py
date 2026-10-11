"""Streaming exhaustive scans over all connected graphs of a given order.

Graphs come from nauty's ``geng`` as graph6 lines; they are decoded in bulk
with numpy and the Laplacian energies of every graph and of all its
vertex-deleted subgraphs are obtained from batched ``numpy.linalg.eigvalsh``.
No per-graph Python object is created in the hot loop, which is what makes
order ten (11 716 571 connected graphs) feasible in minutes.
"""
import shutil
import subprocess
import time

import numpy as np


def find_geng():
    """Locate the nauty ``geng`` executable (Sage's copy, ``geng`` or ``nauty-geng``)."""
    try:
        from sage.features.nauty import NautyExecutable
        return str(NautyExecutable("geng").absolute_filename())
    except Exception:
        pass
    for name in ("geng", "nauty-geng"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("geng not found: install nauty (e.g. `apt install nauty`) "
                       "or run inside SageMath")


class Batch:
    """Decoder and batched energy evaluator for graphs of a fixed order ``n``."""

    def __init__(self, n):
        self.n = n = int(n)
        self.nb = (n*(n-1)//2 + 5)//6
        self.line = 1 + self.nb + 1
        self.nbits = n*(n-1)//2
        rows, cols = [], []
        for j in range(1, n):
            for i in range(j):
                rows.append(i)
                cols.append(j)
        self.rows = np.asarray(rows, dtype=np.intp)
        self.cols = np.asarray(cols, dtype=np.intp)
        self.diag = np.arange(n, dtype=np.intp)
        self.diag1 = np.arange(max(n - 1, 0), dtype=np.intp)
        self.keep = [np.asarray([u for u in range(n) if u != v], dtype=np.intp)
                     for v in range(n)]

    def decode(self, buf):
        """bytes of whole graph6 lines -> (adjacency tensor, graph6 rows)."""
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, self.line)
        data = arr[:, 1:1 + self.nb] - np.uint8(63)
        bits = (np.unpackbits(data, axis=1).reshape(-1, self.nb, 8)[:, :, 2:]
                .reshape(-1, self.nb*6)[:, :self.nbits])
        A = np.zeros((bits.shape[0], self.n, self.n), dtype=np.float64)
        A[:, self.rows, self.cols] = bits
        A[:, self.cols, self.rows] = bits
        return A, arr[:, :self.line - 1]

    def energies(self, A):
        """Return LE(G), LE(G-v) for every v, the degrees, and the size m."""
        n = self.n
        deg = A.sum(axis=2)
        L = -A.copy()
        L[:, self.diag, self.diag] = deg
        le = np.abs(np.linalg.eigvalsh(L) - (deg.sum(axis=1)/n)[:, None]).sum(axis=1)
        le_del = np.zeros((A.shape[0], n))
        for v in range(n if n > 1 else 0):
            Av = A[:, self.keep[v]][:, :, self.keep[v]]
            dv = Av.sum(axis=2)
            Lv = -Av
            Lv[:, self.diag1, self.diag1] = dv
            le_del[:, v] = np.abs(np.linalg.eigvalsh(Lv)
                                  - (dv.sum(axis=1)/(n - 1))[:, None]).sum(axis=1)
        return le, le_del, deg, deg.sum(axis=1)/2.0


def stream(n, chunk=100000, resmod=None, progress=True):
    """Yield (batch, A, graph6_rows, LE, LE_del, deg, m) for all connected graphs of order n."""
    geng = find_geng()
    bt = Batch(n)
    args = [geng, "-c", "-q", str(int(n))]
    if resmod:
        args.append(str(resmod))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1 << 20)
    block = int(chunk)*bt.line
    tail, done, t0, last = b"", 0, time.time(), 0
    try:
        while True:
            raw = proc.stdout.read(block)
            if not raw:
                break
            raw = tail + raw
            use = (len(raw)//bt.line)*bt.line
            tail, buf = raw[use:], raw[:use]
            if not buf:
                continue
            A, g6 = bt.decode(buf)
            le, le_del, deg, m = bt.energies(A)
            done += A.shape[0]
            if progress and done - last >= 20*int(chunk):
                last = done
                print("      ... order %d : %d graphs  (%.0f s)" % (n, done, time.time() - t0), flush=True)
            yield bt, A, g6, le, le_del, deg, m
    finally:
        proc.stdout.close()
        proc.wait()
    if tail:
        raise RuntimeError("truncated geng output")
