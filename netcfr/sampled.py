"""Weaker-feedback learners used for comparison with full-width counterfactual updates (ER-40).

* pg_exact      softmax (logistic) policy gradient with the exact gradient
                (a centralized exact critic): d u_i / d theta = reach * p (1-p) (Q_a - Q_b), lr 5.0
* reinforce     decentralized score-function gradient: batch of 64 sampled deals, each player
                uses only its own realized total payoff, per-player batch-mean baseline, lr 0.5
* bandit        outcome-sampling EXP3 at every information set: one sampled deal per iteration,
                exploration 0.1, eta 0.02, importance-weighted realized payoff (stance: total
                payoff; response: payoff on that edge)
"""

from __future__ import annotations

import time

import numpy as np

from .game import NSP
from .solvers import Averager, RunLog, _logistic


def pg_exact(g: NSP, T: int = 3000, lr: float = 5.0, log_every: int = 50):
    tB = np.zeros((g.n, g.M)); tL = np.zeros((2 * g.E, g.M))
    avg = Averager(g)
    log = RunLog(); t0 = time.perf_counter()
    for t in range(1, T + 1):
        beta, gamma = _logistic(tB), _logistic(tL)
        avg.add(beta, gamma, 1.0)
        v = g.values(beta, gamma)
        tB += lr * beta * (1 - beta) * (v.B - v.C)
        tL += lr * (1 - beta[g.src]) * gamma * (1 - gamma) * (v.cL - v.cF)
        if log_every and (t == 1 or t % log_every == 0 or t == T):
            cur = (_logistic(tB), _logistic(tL))
            log.t.append(t); log.nc_last.append(g.nashconv(*cur)[0])
            log.nc_avg.append(g.nashconv(*avg.profile())[0])
    log.seconds = time.perf_counter() - t0
    return dict(last=(_logistic(tB), _logistic(tL)), avg=avg.profile(), log=log)


def _edge_payoffs(g: NSP, cards, stance, call_dir):
    """Realized payoff of each endpoint on each undirected edge for a batch of deals.
    cards, stance: (B, n); call_dir: (B, 2E) sampled call decision of each directed-edge owner.
    Returns pay_i, pay_j of shape (B, E) for edge (i, j)."""
    E = g.E
    i, j = g.src[:E], g.dst[:E]
    w, b = g.stake_edges[None, :], g.bet_edges[None, :]
    ci, cj = cards[:, i], cards[:, j]
    si, sj = stance[:, i], stance[:, j]
    sg = np.sign(ci - cj)
    same = si == sj
    pay_i = np.where(same, w * sg * np.where(si == 1, 1 + b, 1.0), 0.0)
    # i bets, j responds (owner of directed edge E + e is j)
    call_j = call_dir[:, E:]
    ib = (si == 1) & (sj == 0)
    pay_i = np.where(ib, (1 - call_j) * w + call_j * (1 + b) * w * sg, pay_i)
    # j bets, i responds (owner of directed edge e is i)
    call_i = call_dir[:, :E]
    jb = (si == 0) & (sj == 1)
    pay_i = np.where(jb, -((1 - call_i) * w + call_i * (1 + b) * w * (-sg)), pay_i)
    return pay_i, -pay_i


def reinforce(g: NSP, T: int = 3000, batch: int = 64, lr: float = 0.5, seed: int = 0, log_every: int = 50):
    rng = np.random.default_rng(seed)
    tB = np.zeros((g.n, g.M)); tL = np.zeros((2 * g.E, g.M))
    avg = Averager(g)
    log = RunLog(); t0 = time.perf_counter()
    E = g.E
    for t in range(1, T + 1):
        beta, gamma = _logistic(tB), _logistic(tL)
        avg.add(beta, gamma, 1.0)
        cards = rng.integers(0, g.M, size=(batch, g.n))
        pb = beta[np.arange(g.n)[None, :], cards]
        stance = (rng.random((batch, g.n)) < pb).astype(int)
        owner_card = cards[:, g.src]
        pg = gamma[np.arange(2 * E)[None, :], owner_card]
        call = (rng.random((batch, 2 * E)) < pg).astype(float)
        pay_i, pay_j = _edge_payoffs(g, cards, stance, call)
        R = np.zeros((batch, g.n))
        np.add.at(R.T, g.src[:E], pay_i.T)
        np.add.at(R.T, g.dst[:E], pay_j.T)
        A = R - R.mean(axis=0, keepdims=True)
        gB = np.zeros_like(tB)
        np.add.at(gB, (np.broadcast_to(np.arange(g.n), cards.shape), cards), (stance - pb) * A)
        # response decisions only matter when owner checked and partner bet
        active = (stance[:, g.src] == 0) & (stance[:, g.dst] == 1)
        gL = np.zeros_like(tL)
        contrib = np.where(active, (call - pg) * A[:, g.src], 0.0)
        np.add.at(gL, (np.broadcast_to(np.arange(2 * E), owner_card.shape), owner_card), contrib)
        tB += lr * gB / batch
        tL += lr * gL / batch
        if log_every and (t == 1 or t % log_every == 0 or t == T):
            log.t.append(t); log.nc_last.append(g.nashconv(_logistic(tB), _logistic(tL))[0])
            log.nc_avg.append(g.nashconv(*avg.profile())[0])
    log.seconds = time.perf_counter() - t0
    return dict(last=(_logistic(tB), _logistic(tL)), avg=avg.profile(), log=log)


def bandit(g: NSP, T: int = 60000, eta: float = 0.02, explore: float = 0.1, seed: int = 3,
           log_every: int = 1000):
    rng = np.random.default_rng(seed)
    SB = np.zeros((g.n, g.M)); SC = np.zeros((g.n, g.M))
    SL = np.zeros((2 * g.E, g.M)); SF = np.zeros((2 * g.E, g.M))
    avg = Averager(g)
    log = RunLog(); t0 = time.perf_counter()
    E = g.E
    nodes, dirs = np.arange(g.n), np.arange(2 * E)
    for t in range(1, T + 1):
        beta, gamma = _logistic(eta * (SB - SC)), _logistic(eta * (SL - SF))
        avg.add(beta, gamma, 1.0)
        cards = rng.integers(0, g.M, size=g.n)
        qb = (1 - explore) * beta[nodes, cards] + explore * 0.5
        stance = (rng.random(g.n) < qb).astype(int)
        oc = cards[g.src]
        qg = (1 - explore) * gamma[dirs, oc] + explore * 0.5
        call = (rng.random(2 * E) < qg).astype(float)
        pay_i, pay_j = _edge_payoffs(g, cards[None, :], stance[None, :], call[None, :])
        pay_i, pay_j = pay_i[0], pay_j[0]
        R = np.zeros(g.n)
        np.add.at(R, g.src[:E], pay_i); np.add.at(R, g.dst[:E], pay_j)
        # stance: importance-weighted total payoff for the chosen stance
        prob = np.where(stance == 1, qb, 1 - qb)
        est = R / prob
        np.add.at(SB, (nodes[stance == 1], cards[stance == 1]), est[stance == 1])
        np.add.at(SC, (nodes[stance == 0], cards[stance == 0]), est[stance == 0])
        # responses: realized payoff on that edge, only where the response was reached
        edge_pay = np.concatenate([pay_i, pay_j])            # owner's payoff on directed edge
        active = (stance[g.src] == 0) & (stance[g.dst] == 1)
        pr = np.where(call == 1, qg, 1 - qg)
        est_r = np.where(active, edge_pay / pr, 0.0)
        m_call, m_fold = active & (call == 1), active & (call == 0)
        np.add.at(SL, (dirs[m_call], oc[m_call]), est_r[m_call])
        np.add.at(SF, (dirs[m_fold], oc[m_fold]), est_r[m_fold])
        if log_every and (t == 1 or t % log_every == 0 or t == T):
            ba, ga = avg.profile()
            log.t.append(t); log.nc_avg.append(g.nashconv(ba, ga)[0])
            log.nc_last.append(g.nashconv(_logistic(eta * (SB - SC)), _logistic(eta * (SL - SF)))[0])
    log.seconds = time.perf_counter() - t0
    return dict(avg=avg.profile(), log=log)
