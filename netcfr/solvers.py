"""Full-width NetCFR and fictitious play.

Variants and their update rules:

  cfr    regret matching, R <- R + r, averaging weight 1
  cfr+   R <- max(R + r, 0), averaging weight t
  dcfr   R <- R + r, then positive R *= t^a/(t^a+1), negative R *= t^b/(t^b+1) (a=1.5, b=0);
         average sums *= ((t-1)/t)^g (g=2), weight 1
  pcfr+  strategy = RM(R + m) with m the last instantaneous regret, R <- max(R + r, 0), weight t^2
  hedge  strategy = logistic(eta_t (R_B - R_C)), eta_t = eta / sqrt(t), weight 1
  omwu   hedge applied to R + m (optimistic prediction), weight 1

All players update simultaneously. Responses are averaged with the owner's reach (probability
of checking), so the average is the behavioural form of the averaged realization plans.

An optional `channel` distorts the neighbour strategies each player receives (distributed
NetCFR with quantized or noisy messages).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .game import NSP

VARIANTS = ("cfr", "cfr+", "dcfr", "pcfr+", "hedge", "omwu")


def _rm(a, b):
    ap, bp = np.maximum(a, 0.0), np.maximum(b, 0.0)
    s = ap + bp
    return np.where(s > 0, ap / np.where(s > 0, s, 1.0), 0.5)


def _logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class RunLog:
    t: list = field(default_factory=list)
    nc_avg: list = field(default_factory=list)
    nc_last: list = field(default_factory=list)
    seconds: float = 0.0
    iter_seconds: float = 0.0

    def as_dict(self):
        return dict(t=self.t, nc_avg=self.nc_avg, nc_last=self.nc_last,
                    seconds=self.seconds, iter_seconds=self.iter_seconds)


class Averager:
    def __init__(self, g: NSP):
        self.sB = np.zeros((g.n, g.M))
        self.sW = 0.0
        self.sG = np.zeros((2 * g.E, g.M))
        self.sGW = np.zeros((2 * g.E, g.M))
        self.src = g.src

    def add(self, beta, gamma, w, discount=1.0):
        if discount != 1.0:
            self.sB *= discount
            self.sW *= discount
            self.sG *= discount
            self.sGW *= discount
        reach = 1.0 - beta[self.src]
        self.sB += w * beta
        self.sW += w
        self.sG += w * reach * gamma
        self.sGW += w * reach

    def profile(self):
        beta = self.sB / self.sW
        gamma = np.where(self.sGW > 1e-300, self.sG / np.where(self.sGW > 1e-300, self.sGW, 1.0), 0.5)
        return beta, gamma


def _strategy(variant, t, eta, RB, RC, RL, RF, mB, mC, mL, mF):
    """Strategy played at iteration t from the regret state."""
    if variant in ("cfr", "cfr+", "dcfr"):
        return _rm(RB, RC), _rm(RL, RF)
    if variant == "pcfr+":
        return _rm(RB + mB, RC + mC), _rm(RL + mL, RF + mF)
    et = eta / np.sqrt(t)
    if variant == "hedge":
        return _logistic(et * (RB - RC)), _logistic(et * (RL - RF))
    return (_logistic(et * ((RB + mB) - (RC + mC))),
            _logistic(et * ((RL + mL) - (RF + mF))))


def netcfr(g: NSP, variant: str = "cfr+", T: int = 1000, eta: float = 1.0,
           dcfr_params=(1.5, 0.0, 2.0), log_every: int = 50, channel=None,
           track_last: bool = True, eval_avg: bool = True, stop_nc_per_agent: float | None = None,
           check_every: int = 10, snapshots=()):
    """Run T iterations. `snapshots` is an iterable of iterations at which the per-agent
    exploitability gaps (best-response value - value) of the average and current profiles are
    recorded in the returned dict under 'snapshots'."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    n, M, E2 = g.n, g.M, 2 * g.E
    RB = np.zeros((n, M)); RC = np.zeros((n, M))
    RL = np.zeros((E2, M)); RF = np.zeros((E2, M))
    mB = np.zeros((n, M)); mC = np.zeros((n, M)); mL = np.zeros((E2, M)); mF = np.zeros((E2, M))
    avg = Averager(g)
    log = RunLog()
    a_d, b_d, g_d = dcfr_params
    t0 = time.perf_counter()
    it_time = 0.0
    stopped_at = None
    snap_at = set(snapshots)
    snaps = []
    for t in range(1, T + 1):
        ti = time.perf_counter()
        # ---- strategy from regrets
        beta, gamma = _strategy(variant, t, eta, RB, RC, RL, RF, mB, mC, mL, mF)
        # ---- counterfactual values (optionally from distorted neighbour messages)
        partner = channel(g, beta, gamma) if channel is not None else None
        v = g.values(beta, gamma, partner)
        vs = beta * v.B + (1.0 - beta) * v.C
        rB, rC = v.B - vs, v.C - vs
        resp = gamma * v.cL + (1.0 - gamma) * v.cF
        rL, rF = v.cL - resp, v.cF - resp
        # ---- regret update
        if variant in ("cfr", "hedge", "omwu"):
            RB += rB; RC += rC; RL += rL; RF += rF
        elif variant in ("cfr+", "pcfr+"):
            RB = np.maximum(RB + rB, 0); RC = np.maximum(RC + rC, 0)
            RL = np.maximum(RL + rL, 0); RF = np.maximum(RF + rF, 0)
        else:  # dcfr
            pos, neg = t**a_d / (t**a_d + 1), t**b_d / (t**b_d + 1)
            for R, r in ((RB, rB), (RC, rC), (RL, rL), (RF, rF)):
                R += r
                R *= np.where(R > 0, pos, neg)
        if variant in ("pcfr+", "omwu"):
            mB, mC, mL, mF = rB, rC, rL, rF
        # ---- averaging
        if variant == "cfr+":
            avg.add(beta, gamma, float(t))
        elif variant == "pcfr+":
            avg.add(beta, gamma, float(t) ** 2)
        elif variant == "dcfr":
            avg.add(beta, gamma, 1.0, discount=((t - 1) / t) ** g_d if t > 1 else 1.0)
        else:
            avg.add(beta, gamma, 1.0)
        it_time += time.perf_counter() - ti
        if t in snap_at:
            ba, ga = avg.profile()
            _, ua, bra = g.nashconv(ba, ga)
            cur = _strategy(variant, t + 1, eta, RB, RC, RL, RF, mB, mC, mL, mF)
            _, uc, brc = g.nashconv(*cur)
            snaps.append(dict(t=t, gap_avg=(bra - ua).tolist(), gap_last=(brc - uc).tolist()))
        # ---- logging (not timed)
        do_log = log_every and (t == 1 or t % log_every == 0 or t == T)
        do_check = stop_nc_per_agent is not None and t % check_every == 0
        if do_log or do_check:
            ba, ga = avg.profile()
            nc_a = g.nashconv(ba, ga)[0] if (eval_avg or do_check) else float("nan")
            if do_log:
                log.t.append(t)
                log.nc_avg.append(nc_a)
                if track_last:   # current policy = strategy after the t-th update
                    cur = _strategy(variant, t + 1, eta, RB, RC, RL, RF, mB, mC, mL, mF)
                    log.nc_last.append(g.nashconv(*cur)[0])
                else:
                    log.nc_last.append(float("nan"))
            if do_check and nc_a / g.n <= stop_nc_per_agent:
                stopped_at = t
                break
    log.seconds = time.perf_counter() - t0
    log.iter_seconds = it_time / (stopped_at or T)
    ba, ga = avg.profile()
    last = _strategy(variant, (stopped_at or T) + 1, eta, RB, RC, RL, RF, mB, mC, mL, mF)
    return dict(avg=(ba, ga), last=last, log=log, stopped_at=stopped_at, snapshots=snaps)


def fictitious_play(g: NSP, T: int = 1000, log_every: int = 50):
    """Discrete FP: best respond to the average profile. Uniform start with
    sB = 1/2, sW = 1, sG = 1/4, sGW = 1/2."""
    avg = Averager(g)
    avg.sB[:] = 0.5; avg.sW = 1.0; avg.sG[:] = 0.25; avg.sGW[:] = 0.5
    log = RunLog()
    t0 = time.perf_counter()
    bet = call = None
    for t in range(1, T + 1):
        ba, ga = avg.profile()
        bet, call = g.best_response(ba, ga)
        avg.add(bet, call, 1.0)
        if log_every and (t == 1 or t % log_every == 0 or t == T):
            ba, ga = avg.profile()
            log.t.append(t)
            log.nc_avg.append(g.nashconv(ba, ga)[0])
            log.nc_last.append(g.nashconv(bet, call)[0])
    log.seconds = time.perf_counter() - t0
    return dict(avg=avg.profile(), last=(bet, call), log=log)


# ------------------------------------------------------------------ message channels
def quantizer(q: int):
    """round(x L) / L with L = 2^q - 1 applied to every received probability."""
    L = 2**q - 1

    def ch(g: NSP, beta, gamma):
        return np.round(beta[g.dst] * L) / L, np.round(gamma[g.rev] * L) / L
    ch.name = f"q{q}"
    return ch


def laplace(sigma: float, seed: int = 23):
    """clip(x + Laplace(0, sigma), 0, 1), independent per directed message and entry."""
    rng = np.random.default_rng(seed)

    def ch(g: NSP, beta, gamma):
        bj, gj = beta[g.dst], gamma[g.rev]
        return (np.clip(bj + rng.laplace(0, sigma, bj.shape), 0, 1),
                np.clip(gj + rng.laplace(0, sigma, gj.shape), 0, 1))
    ch.name = f"laplace{sigma}"
    return ch
