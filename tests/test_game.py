"""Correctness tests: exact small-game equilibria, brute-force enumeration, zero-sum identity,
benchmark instance regeneration, LP ground truth, and the message-distortion lemma."""

import itertools
from fractions import Fraction as Fr

import networkx as nx
import numpy as np
import pytest

from netcfr import instances as ins
from netcfr.game import NSP
from netcfr.lp import Layout, security_value, solve_ne


def edge_game(b, K=3):
    return NSP(2, [(0, 1)], [b], [1.0], K)


# ------------------------------------------------------------------ single edge game E(1), K=3
def test_edge_game_counterfactual_values():
    g = edge_game(1.0)
    beta = np.array([[0, 2 / 3, 2 / 3]] * 2)
    gamma = np.array([[0, 0, 1.0]] * 2)
    v = g.values(beta, gamma)
    d = 0  # directed edge 0 -> 1, owner 0
    cC0 = v.C[0] - (gamma[d] * v.cL[d] + (1 - gamma[d]) * v.cF[d])
    np.testing.assert_allclose(cC0, np.array([-2, 2, 4]) / 27, atol=1e-12)
    np.testing.assert_allclose(v.cF[d], np.array([-4, -4, -4]) / 27, atol=1e-12)
    np.testing.assert_allclose(v.cL[d], np.array([-8, -4, 4]) / 27, atol=1e-12)
    np.testing.assert_allclose(v.Cbr[0], np.array([-6, -2, 8]) / 27, atol=1e-12)
    np.testing.assert_allclose(v.B[0], np.array([-6, -2, 8]) / 27, atol=1e-12)
    nc, u, _ = g.nashconv(beta, gamma)
    assert abs(nc) < 1e-12 and np.allclose(u, 0)


# ------------------------------------------------------------------ path with bet ratios 1 and 3
def path_1_3():
    return NSP(3, [(0, 1), (1, 2)], [1.0, 3.0], [1.0, 1.0], 3)


def test_path_equilibrium():
    g = path_1_3()
    beta = np.array([[1 / 9, 2 / 3, 1], [2 / 9, 0, 2 / 3], [1 / 3, 0, 5 / 9]])
    # directed edges: 0:(0->1) 1:(1->2) 2:(1->0) 3:(2->1)
    gamma = np.array([[0, 0, 0.5], [0, 0, 1], [0, 2 / 3, 1], [0, 0, 1]])
    nc, u, _ = g.nashconv(beta, gamma)
    assert abs(nc) < 1e-12
    np.testing.assert_allclose(u, [2 / 81, -2 / 81, 0], atol=1e-12)


def test_security_value_center_path():
    assert security_value(path_1_3(), 1) == pytest.approx(-2 / 81, abs=1e-9)


@pytest.mark.parametrize("ratios,center", [
    ((1, 3), -0.0247), ((1, 2, 3, 4), -0.0802), ((1, 1, 1, 1, 4), -0.1235), ((1, 2, 3, 4, 5, 6), -0.1296)])
def test_star_center_security_and_payoff(ratios, center):
    k = len(ratios)
    g = NSP(k + 1, [(0, i + 1) for i in range(k)], list(map(float, ratios)), [1.0] * k, 3)
    assert security_value(g, 0) == pytest.approx(center, abs=5e-5)
    sol = solve_ne(g)
    nc, u, _ = g.nashconv(sol["beta"], sol["gamma"])
    assert nc < 1e-7
    assert u[0] == pytest.approx(center, abs=5e-5)


def test_alternating_cycle_security():
    g = NSP(6, [(i, (i + 1) % 6) for i in range(6)], [1.0, 3.0] * 3, [1.0] * 6, 3)
    for i in range(6):
        assert security_value(g, i) == pytest.approx(-2 / 81, abs=1e-9)


# ------------------------------------------------------------------ brute force
def brute_utilities(g, beta, gamma):
    """Exact expectation by enumerating cards and action outcomes (tiny graphs only)."""
    n, M = g.n, g.M
    u = np.zeros(n)
    dir_of = {(int(g.src[d]), int(g.dst[d])): d for d in range(2 * g.E)}
    for cards in itertools.product(range(M), repeat=n):
        pc = 1.0 / M**n
        for stances in itertools.product([0, 1], repeat=n):
            ps = np.prod([beta[i, cards[i]] if s else 1 - beta[i, cards[i]] for i, s in enumerate(stances)])
            if ps == 0:
                continue
            for e, (i, j) in enumerate(g.edges):
                w, b = g.stake_edges[e], g.bet_edges[e]
                sg = np.sign(cards[i] - cards[j])
                si, sj = stances[i], stances[j]
                if si == sj:
                    pay = w * sg * (1 + b if si else 1)
                    u[i] += pc * ps * pay; u[j] -= pc * ps * pay
                else:
                    bettor, resp = (i, j) if si else (j, i)
                    gcall = gamma[dir_of[(resp, bettor)], cards[resp]]
                    sgb = np.sign(cards[bettor] - cards[resp])
                    pay_b = (1 - gcall) * w + gcall * (1 + b) * w * sgb
                    u[bettor] += pc * ps * pay_b; u[resp] -= pc * ps * pay_b
    return u


def test_values_match_brute_force_triangle():
    rng = np.random.default_rng(0)
    g = NSP(3, [(0, 1), (1, 2), (0, 2)], [0.5, 2.0, 4.0], [1.0, 2.0, 0.5], 3)
    beta, gamma = rng.random((3, 3)), rng.random((6, 3))
    u, _ = g.utilities(beta, gamma)
    np.testing.assert_allclose(u, brute_utilities(g, beta, gamma), atol=1e-12)
    assert abs(u.sum()) < 1e-12


def test_best_response_value_matches_pure_enumeration():
    rng = np.random.default_rng(1)
    g = NSP(3, [(0, 1), (1, 2)], [1.0, 3.0], [1.0, 1.0], 3)
    beta, gamma = rng.random((3, 3)), rng.random((4, 3))
    _, br = g.utilities(beta, gamma)
    # player 1 (degree 2): 2^(M(1+deg)) = 512 pure strategies
    d_own = [d for d in range(4) if g.src[d] == 1]
    best = -np.inf
    for bits in itertools.product([0, 1], repeat=3 * 3):
        b2, gm = beta.copy(), gamma.copy()
        b2[1] = bits[:3]
        gm[d_own[0]] = bits[3:6]
        gm[d_own[1]] = bits[6:9]
        best = max(best, g.utilities(b2, gm)[0][1])
    assert br[1] == pytest.approx(best, abs=1e-12)


# ------------------------------------------------------------------ instance reproduction
@pytest.mark.parametrize("name,maker,expect", [
    ("ER-40", ins.er40, dict(n=40, edges=77, M=3, max_degree=9, isolated=0,
                              ratio_counts={1.0: 46, 3.0: 31}, sequences=1204, treeplex_rows=622)),
    ("BA-100", ins.ba100, dict(n=100, edges=196, M=5, max_degree=29, isolated=0,
                              ratio_counts={0.5: 51, 1.0: 62, 2.0: 48, 4.0: 35},
                              sequences=5020, treeplex_rows=2560)),
    ("ER-100", ins.er100_comm, dict(n=100, edges=192, M=3)),
])
def test_benchmark_instances_regenerate_exactly(name, maker, expect):
    s = maker().summary()
    for k, v in expect.items():
        assert s[k] == v, (name, k, s[k], v)


def test_e03_edge_counts():
    assert nx.fast_gnp_random_graph(1000, 6 / 999, seed=7).number_of_edges() == 3077
    assert nx.fast_gnp_random_graph(10000, 6 / 9999, seed=7).number_of_edges() == 30222


def test_lp_solves_ba100_exactly():
    g = ins.ba100()
    lay = Layout(g)
    assert lay.Nx == 5020 and lay.Nz == 2560
    sol = solve_ne(g)
    nc, u, _ = g.nashconv(sol["beta"], sol["gamma"])
    assert nc < 1e-7 and abs(sol["obj"]) < 1e-7


# ------------------------------------------------------------------ message-distortion lemma
@pytest.mark.parametrize("eps", [0.01, 0.07, 0.3])
def test_distortion_lemma_perceived_utility_within_2_eps_delta(eps):
    """|u~_i(y) - u_i(y)| <= 2 eps Delta_i for every own strategy y when every received
    probability is perturbed by at most eps."""
    rng = np.random.default_rng(2)
    g = ins.er40()
    beta, gamma = rng.random((g.n, g.M)), rng.random((2 * g.E, g.M))
    bj, gj = beta[g.dst], gamma[g.rev]
    pb = np.clip(bj + rng.uniform(-eps, eps, bj.shape), 0, 1)
    pg = np.clip(gj + rng.uniform(-eps, eps, gj.shape), 0, 1)
    D = g.delta()
    for trial in range(40):
        y_beta, y_gamma = rng.random((g.n, g.M)), rng.random((2 * g.E, g.M))
        if trial % 2:   # pure own strategies too (the bound must hold for every plan)
            y_beta, y_gamma = np.round(y_beta), np.round(y_gamma)
        true = g.utilities(y_beta, y_gamma, g.values(y_beta, y_gamma, (bj, gj)))[0]
        pert = g.utilities(y_beta, y_gamma, g.values(y_beta, y_gamma, (pb, pg)))[0]
        assert np.all(np.abs(pert - true) <= 2 * eps * D + 1e-12)


def test_bits_formula_er100():
    g = ins.er100_comm()
    assert [g.bits_per_iteration(q) for q in (1, 2, 3, 4, 6, 8)] == [2304 * q for q in (1, 2, 3, 4, 6, 8)]


def test_quantized_run_satisfies_distortion_bound():
    """End-to-end: NashConv of the uniform average under 3-bit messages is below the
    distortion bound (perceived average regret + 4 eps sum Delta)."""
    from netcfr.solvers import netcfr, quantizer
    g = ins.er100_comm()
    q = 3
    eps = 1 / (2 * (2**q - 1))
    r = netcfr(g, "cfr", T=300, log_every=0, track_last=False, channel=quantizer(q))
    nc = g.nashconv(*r["avg"])[0]
    assert nc <= 4 * eps * g.delta().sum()     # regret term is non-negative, so this is implied


def test_stochastic_rounding_is_unbiased_and_on_grid():
    from netcfr.solvers import stochastic_quantizer
    g = ins.er100_comm()
    rng = np.random.default_rng(5)
    beta, gamma = rng.random((g.n, g.M)), rng.random((2 * g.E, g.M))
    ch = stochastic_quantizer(2, seed=1)
    L = 3
    acc_b = np.zeros_like(beta[g.dst]); acc_v = np.zeros((g.n, g.M))
    true_v = g.values(beta, gamma).B
    N = 4000
    for _ in range(N):
        pb, pg = ch(g, beta, gamma)
        assert np.allclose(pb * L, np.round(pb * L)) and pb.min() >= 0 and pb.max() <= 1
        acc_b += pb
        acc_v += g.values(beta, gamma, (pb, pg)).B
    assert np.abs(acc_b / N - beta[g.dst]).max() < 0.03          # E[Q(x)] = x
    assert np.abs(acc_v / N - true_v).max() < 0.01               # perceived values unbiased


def test_error_feedback_running_sum_tracks_truth():
    from netcfr.solvers import error_feedback_quantizer
    g = ins.er100_comm()
    rng = np.random.default_rng(7)
    ch = error_feedback_quantizer(1)
    tb = np.zeros((2 * g.E, g.M)); rb = np.zeros_like(tb)
    for _ in range(500):
        beta, gamma = rng.random((g.n, g.M)), rng.random((2 * g.E, g.M))
        pb, _ = ch(g, beta, gamma)
        assert set(np.unique(pb)) <= {0.0, 1.0}
        tb += beta[g.dst]; rb += pb
    assert np.abs(tb - rb).max() <= 1.0 + 1e-9          # residual stays bounded by the grid step
