"""Exact centralized solution of NSP by one sequence-form LP (zero-sum polymatrix structure),
plus security (maxmin) values. Used as ground truth and as the centralized baseline.

Sequences of player i (local indices): 0 = root; C(o) = 1+2o, B(o) = 2+2o;
F(o,l) = 1 + 2M + 2(o*deg + l), L(o,l) = F(o,l) + 1, for local edge index l.
Treeplex: x_root = 1; x_C(o) + x_B(o) = x_root; x_F(o,l) + x_L(o,l) = x_C(o).

Payoff blocks of A_ij, S = w sgn(o-o')/M^2, J = w/M^2:
    (C,C) S   (B,B) (1+b)S   (B, F_j) +J   (B, L_j) (1+b)S   (F_i, B) -J   (L_i, B) (1+b)S
NE <=> min sum_i z_i[root]  s.t.  E_i^T z_i >= sum_j A_ij x_j,  E_i x_i = e_i,  x >= 0,
whose optimum is 0 because sum_i u_i == 0.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .game import NSP


class Layout:
    def __init__(self, g: NSP):
        self.g = g
        M = g.M
        self.local = {}                    # directed edge d -> local index l at its owner
        counts = np.zeros(g.n, dtype=int)
        for d in range(2 * g.E):
            i = int(g.src[d])
            self.local[d] = counts[i]
            counts[i] += 1
        self.deg = counts
        self.nseq = 1 + 2 * M + 2 * M * counts
        self.nrow = 1 + M + M * counts
        self.xo = np.concatenate([[0], np.cumsum(self.nseq)[:-1]])
        self.zo = np.concatenate([[0], np.cumsum(self.nrow)[:-1]])
        self.Nx, self.Nz = int(self.nseq.sum()), int(self.nrow.sum())

    def C(self, i, o): return self.xo[i] + 1 + 2 * o
    def B(self, i, o): return self.xo[i] + 2 + 2 * o
    def F(self, i, o, l): return self.xo[i] + 1 + 2 * self.g.M + 2 * (o * self.deg[i] + l)
    def L(self, i, o, l): return self.F(i, o, l) + 1

    def treeplex(self):
        """Sparse E (Nz x Nx) and e (Nz)."""
        g, M = self.g, self.g.M
        r, c, v = [], [], []
        e = np.zeros(self.Nz)
        for i in range(g.n):
            z0 = self.zo[i]
            r.append(z0); c.append(self.xo[i]); v.append(1.0); e[z0] = 1.0
            for o in range(M):
                row = z0 + 1 + o
                r += [row, row, row]; c += [self.C(i, o), self.B(i, o), self.xo[i]]; v += [1, 1, -1]
                for l in range(self.deg[i]):
                    row = z0 + 1 + M + o * self.deg[i] + l
                    r += [row, row, row]
                    c += [self.F(i, o, l), self.L(i, o, l), self.C(i, o)]
                    v += [1, 1, -1]
        return sp.csr_matrix((v, (r, c)), shape=(self.Nz, self.Nx)), e

    def payoff(self):
        """Sparse A (Nx x Nx) with u_i = x_i^T (A x) restricted to i's rows."""
        g, M = self.g, self.g.M
        o = np.arange(M)
        sgn = np.sign(o[:, None] - o[None, :])
        r, c, v = [], [], []
        for d in range(2 * g.E):
            i, j = int(g.src[d]), int(g.dst[d])
            li, lj = self.local[d], self.local[int(g.rev[d])]
            w, b = float(g.w[d, 0]), float(g.b[d, 0])
            S = w * sgn / M**2
            Jv = w / M**2
            for a in range(M):
                for bb in range(M):
                    s = S[a, bb]
                    if s != 0:
                        r += [self.C(i, a), self.B(i, a), self.B(i, a), self.L(i, a, li)]
                        c += [self.C(j, bb), self.B(j, bb), self.L(j, bb, lj), self.B(j, bb)]
                        v += [s, (1 + b) * s, (1 + b) * s, (1 + b) * s]
                    r += [self.B(i, a), self.F(i, a, li)]
                    c += [self.F(j, bb, lj), self.B(j, bb)]
                    v += [Jv, -Jv]
        return sp.csr_matrix((v, (r, c)), shape=(self.Nx, self.Nx))

    def to_behaviour(self, x):
        g, M = self.g, self.g.M
        beta = np.zeros((g.n, M))
        gamma = np.full((2 * g.E, M), 0.5)
        for i in range(g.n):
            for o in range(M):
                cb = x[self.C(i, o)] + x[self.B(i, o)]
                beta[i, o] = x[self.B(i, o)] / cb if cb > 1e-12 else 0.5
        for d in range(2 * g.E):
            i, l = int(g.src[d]), self.local[d]
            for o in range(M):
                den = x[self.C(i, o)]
                if den > 1e-12:
                    gamma[d, o] = x[self.L(i, o, l)] / den
        return np.clip(beta, 0, 1), np.clip(gamma, 0, 1)


def solve_ne(g: NSP):
    """Exact NE by one LP (HiGHS). Returns dict(beta, gamma, seconds, nvars, nnz, obj)."""
    lay = Layout(g)
    E, e = lay.treeplex()
    A = lay.payoff()
    # rows of player i in A: mask by blocks. Constraint: sum_j A_ij x_j - E_i^T z_i <= 0
    Et = E.T.tocsr()                      # Nx x Nz, block-diagonal per player
    A_ub = sp.hstack([A, -Et]).tocsr()
    b_ub = np.zeros(lay.Nx)
    A_eq = sp.hstack([E, sp.csr_matrix((lay.Nz, lay.Nz))]).tocsr()
    c = np.zeros(lay.Nx + lay.Nz)
    c[lay.Nx + lay.zo] = 1.0
    bounds = [(0, None)] * lay.Nx + [(None, None)] * lay.Nz
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=e, bounds=bounds, method="highs")
    secs = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(res.message)
    beta, gamma = lay.to_behaviour(res.x[: lay.Nx])
    return dict(beta=beta, gamma=gamma, seconds=secs, obj=float(res.fun),
                nvars=lay.Nx + lay.Nz, nnz=int(A_ub.nnz + A_eq.nnz))


def security_value(g: NSP, i: int):
    """maxmin value of player i: max_{x_i} sum_{j in N(i)} min_{y_j} x_i^T A_ij y_j."""
    M = g.M
    o = np.arange(M)
    sgn = np.sign(o[:, None] - o[None, :])
    ds = [d for d in range(2 * g.E) if int(g.src[d]) == i]
    k = len(ds)
    nx_i = 1 + 2 * M + 2 * M * k
    Cx = lambda a: 1 + 2 * a
    Bx = lambda a: 2 + 2 * a
    Fx = lambda a, l: 1 + 2 * M + 2 * (a * k + l)
    Lx = lambda a, l: Fx(a, l) + 1
    # neighbour local treeplex (per edge): root, C(o'), B(o'), F(o'), L(o')  -> 1 + 4M sequences
    ny = 1 + 4 * M
    Cy = lambda a: 1 + 4 * a
    By = lambda a: 2 + 4 * a
    Fy = lambda a: 3 + 4 * a
    Ly = lambda a: 4 + 4 * a
    nv = 1 + 2 * M          # rows of neighbour treeplex (dual vars v_j)
    nvar = nx_i + k * nv
    A_ub_rows, b_ub = [], []
    r, c, v = [], [], []
    row = 0
    for l, d in enumerate(ds):
        w, b = float(g.w[d, 0]), float(g.b[d, 0])
        S = w * sgn / M**2
        Jv = w / M**2
        A = np.zeros((nx_i, ny))           # payoff to i: x_i^T A y_j
        for a in range(M):
            for bb in range(M):
                s = S[a, bb]
                A[Cx(a), Cy(bb)] += s
                A[Bx(a), By(bb)] += (1 + b) * s
                A[Bx(a), Ly(bb)] += (1 + b) * s
                A[Lx(a, l), By(bb)] += (1 + b) * s
                A[Bx(a), Fy(bb)] += Jv
                A[Fx(a, l), By(bb)] += -Jv
        # neighbour treeplex Ey (nv x ny)
        Ey = np.zeros((nv, ny))
        Ey[0, 0] = 1
        for a in range(M):
            Ey[1 + a, [Cy(a), By(a), 0]] = [1, 1, -1]
            Ey[1 + M + a, [Fy(a), Ly(a), Cy(a)]] = [1, 1, -1]
        # E_y^T v_j - A^T x_i <= 0   (ny rows)
        block = np.hstack([-A.T, np.zeros((ny, k * nv))])
        block[:, nx_i + l * nv: nx_i + (l + 1) * nv] = Ey.T
        A_ub_rows.append(block)
        b_ub += [0.0] * ny
    A_ub = np.vstack(A_ub_rows) if A_ub_rows else np.zeros((0, nvar))
    # own treeplex
    Ex = np.zeros((1 + M + M * k, nvar))
    Ex[0, 0] = 1
    for a in range(M):
        Ex[1 + a, [Cx(a), Bx(a), 0]] = [1, 1, -1]
        for l in range(k):
            Ex[1 + M + a * k + l, [Fx(a, l), Lx(a, l), Cx(a)]] = [1, 1, -1]
    ex = np.zeros(Ex.shape[0]); ex[0] = 1
    cvec = np.zeros(nvar)
    for l in range(k):
        cvec[nx_i + l * nv] = -1.0         # maximize sum_j v_j[root]
    bounds = [(0, None)] * nx_i + [(None, None)] * (k * nv)
    res = linprog(cvec, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=Ex, b_eq=ex, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun)
