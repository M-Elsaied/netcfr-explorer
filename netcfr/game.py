"""Network Stance Poker (NSP) with edge-local information, i.i.d. uniform cards, no public card.

Rules. Each edge e={i,j} has stake w_e and bet ratio b_e.
Each player i holds a private card o in {0..M-1} (higher is stronger) and chooses ONE stance,
check (C) or bet (B), for all incident edges. On an edge where exactly one endpoint bets, the
other endpoint folds (F) or calls (L) on that edge alone. Payoff to i on e:

    (C,C): w * sgn(o_i - o_j)            (B,B): (1+b) w * sgn
    i bets, j folds: +w                  i bets, j calls: (1+b) w * sgn
    j bets, i folds: -w                  j bets, i calls: (1+b) w * sgn

A behavioural profile is beta (n x M) = P(bet | card) and gamma (2|E| x M), where gamma[d] is
the call probability of the OWNER of directed edge d = (src -> dst) on that edge.

Counterfactual values are chance-and-opponent weighted:
with P(o, o') = 1/M^2 and S = sgn(o - o')/M^2, for directed edge d with partner strategy
(beta_j, gamma_j) the owner's values are

    cB  = w [ (1+b) S beta_j + P ((1-beta_j)(1-gamma_j)) + (1+b) S ((1-beta_j) gamma_j) ]
    cC0 = w S (1 - beta_j)        (check, partner also checks)
    cL  = w (1+b) S beta_j        (check, partner bets, owner calls)
    cF  = -w P beta_j             (check, partner bets, owner folds)

Stance values sum over incident edges; the response information set J_{i,e}^o uses cL, cF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Values:
    B: np.ndarray       # (n, M) value of betting
    C: np.ndarray       # (n, M) value of checking with current responses
    Cbr: np.ndarray     # (n, M) value of checking with best-response responses
    cL: np.ndarray      # (2E, M) call value at response infosets
    cF: np.ndarray      # (2E, M) fold value at response infosets


class NSP:
    def __init__(self, n: int, edges, bets, stakes, K: int):
        self.n = int(n)
        self.M = int(K)
        self.edges = [(int(i), int(j)) for i, j in edges]
        E = len(self.edges)
        self.E = E
        ii = np.array([e[0] for e in self.edges], dtype=np.int64)
        jj = np.array([e[1] for e in self.edges], dtype=np.int64)
        # directed edges: d in [0,E) is i->j, d in [E,2E) is j->i
        self.src = np.concatenate([ii, jj])
        self.dst = np.concatenate([jj, ii])
        self.rev = np.concatenate([np.arange(E, 2 * E), np.arange(0, E)])
        b = np.asarray(bets, dtype=float)
        w = np.asarray(stakes, dtype=float)
        self.b = np.concatenate([b, b])[:, None]
        self.w = np.concatenate([w, w])[:, None]
        self.bet_edges = b
        self.stake_edges = w
        M = self.M
        o = np.arange(M)
        self.Sg = np.sign(o[:, None] - o[None, :]) / M**2      # S[o, o']
        self.P1 = np.full((M, M), 1.0 / M**2)
        # incidence: node i sums the directed edges it owns
        self.inc = sp.csr_matrix(
            (np.ones(2 * E), (self.src, np.arange(2 * E))), shape=(self.n, 2 * E)
        )
        self.deg = np.asarray(self.inc.sum(axis=1)).ravel()

    # ---------------------------------------------------------------- helpers
    def uniform(self):
        return np.full((self.n, self.M), 0.5), np.full((2 * self.E, self.M), 0.5)

    def _S(self, x):   # (2E, M) -> sum_o' S[o,o'] x[d,o']
        return x @ self.Sg.T

    def _P(self, x):
        return x @ self.P1.T

    # ---------------------------------------------------------------- core evaluation
    def values(self, beta, gamma, partner=None) -> Values:
        """Counterfactual values. `partner` optionally overrides the (beta_j, gamma_j) seen on
        each directed edge (used for distorted messages); default is the true profile."""
        if partner is None:
            bj = beta[self.dst]
            gj = gamma[self.rev]
        else:
            bj, gj = partner
        w, b1 = self.w, 1.0 + self.b
        nb = 1.0 - bj
        cB = w * (b1 * self._S(bj) + self._P(nb * (1.0 - gj)) + b1 * self._S(nb * gj))
        cC0 = w * self._S(nb)
        cL = w * b1 * self._S(bj)
        cF = -w * self._P(bj)
        g = gamma
        B = self.inc @ cB
        C = self.inc @ (cC0 + g * cL + (1.0 - g) * cF)
        Cbr = self.inc @ (cC0 + np.maximum(cL, cF))
        return Values(B, C, Cbr, cL, cF)

    def utilities(self, beta, gamma, vals: Values | None = None):
        v = vals or self.values(beta, gamma)
        u = (beta * v.B + (1.0 - beta) * v.C).sum(axis=1)
        br = np.maximum(v.B, v.Cbr).sum(axis=1)
        return u, br

    def nashconv(self, beta, gamma, vals: Values | None = None):
        """Exact NashConv = sum_i (best-response value - value). Returns (total, u, br)."""
        u, br = self.utilities(beta, gamma, vals)
        return float((br - u).sum()), u, br

    def best_response(self, beta, gamma):
        v = self.values(beta, gamma)
        bet = (v.B > v.Cbr).astype(float)
        call = (v.cL > v.cF).astype(float)
        return bet, call

    # ---------------------------------------------------------------- bookkeeping
    def delta(self):
        """Payoff-range constants Delta_i = sum_{e ni i} 2 w_e (1 + b_e)."""
        per_dir = (2.0 * self.w * (1.0 + self.b)).ravel()
        return np.bincount(self.src, weights=per_dir, minlength=self.n)

    def bits_per_iteration(self, q: int) -> int:
        """Point-to-point NetCFR messages: 2 directed messages per edge, each carrying the
        sender's stance vector and its response vector on that edge (2M numbers) at q bits."""
        return 2 * self.E * 2 * self.M * q

    def summary(self):
        return dict(n=self.n, edges=self.E, M=self.M, max_degree=int(self.deg.max()),
                    isolated=int((self.deg == 0).sum()),
                    ratio_counts={float(v): int((self.bet_edges == v).sum())
                                  for v in np.unique(self.bet_edges)},
                    sequences=int(self.n + 2 * self.M * (self.n + 2 * self.E)),
                    treeplex_rows=int(self.n + self.M * (self.n + 2 * self.E)))
