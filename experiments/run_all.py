"""Run all simulations and write results/*.json.

    uv run python experiments/run_all.py                  # everything (~2-3 min)
    uv run python experiments/run_all.py --only network   # one experiment
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netcfr import instances as ins  # noqa: E402
from netcfr.lp import security_value, solve_ne  # noqa: E402
from netcfr.sampled import bandit, pg_exact, reinforce  # noqa: E402
from netcfr.solvers import fictitious_play, laplace, netcfr, quantizer  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def run(label, fn):
    t0 = time.perf_counter()
    print(f"[{label}] ...", flush=True)
    fn()
    print(f"[{label}] done in {time.perf_counter() - t0:.1f}s", flush=True)


def ba100_learners():
    g = ins.ba100()
    lp = solve_ne(g)
    out = dict(instance=g.summary(), lp=dict(seconds=lp["seconds"],
                                              nashconv=g.nashconv(lp["beta"], lp["gamma"])[0]))
    runs = {}
    for v, eta in [("pcfr+", 1), ("cfr+", 1), ("dcfr", 1), ("cfr", 1),
                   ("hedge", 2), ("hedge", 10), ("hedge", 50), ("omwu", 2), ("omwu", 10), ("omwu", 50)]:
        r = netcfr(g, v, T=3000, eta=eta, log_every=50)
        runs[v + (f"_eta{eta}" if v in ("hedge", "omwu") else "")] = r["log"].as_dict()
    runs["fp"] = fictitious_play(g, T=3000, log_every=50)["log"].as_dict()
    out["runs"] = runs
    save("ba100_learners", out)


def er40_learners():
    g = ins.er40()
    runs = {}
    for v, eta in [("cfr+", 1), ("pcfr+", 1), ("dcfr", 1), ("cfr", 1), ("hedge", 10), ("omwu", 10)]:
        r = netcfr(g, v, T=3000, eta=eta, log_every=50)
        runs[v + (f"_eta{eta}" if v in ("hedge", "omwu") else "")] = r["log"].as_dict()
    runs["fp"] = fictitious_play(g, T=3000, log_every=50)["log"].as_dict()
    runs["pg_exact"] = pg_exact(g, T=3000, lr=5.0, log_every=50)["log"].as_dict()
    runs["reinforce"] = reinforce(g, T=3000, batch=64, lr=0.5, seed=0, log_every=50)["log"].as_dict()
    runs["bandit"] = bandit(g, T=60000, eta=0.02, explore=0.1, seed=3, log_every=1000)["log"].as_dict()
    save("er40_learners", dict(instance=g.summary(), runs=runs))


def scaling(sizes=(1000, 10000, 100000)):
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        g = ins.er_scaling(n)
        build = time.perf_counter() - t0
        r = netcfr(g, "cfr+", T=100, log_every=0, track_last=False)
        rows.append(dict(n=n, edges=g.E, build_s=build, s_per_iter=r["log"].iter_seconds,
                         us_per_edge_iter=1e6 * r["log"].iter_seconds / g.E,
                         nashconv_per_agent=g.nashconv(*r["avg"])[0] / n))
        print("   ", rows[-1], flush=True)
    x, y = np.log([r["edges"] for r in rows]), np.log([r["s_per_iter"] for r in rows])
    save("scaling", dict(rows=rows, slope_time_vs_edges=float(np.polyfit(x, y, 1)[0]),
                         machine=platform.processor() or platform.machine()))


def communication():
    g = ins.er100_comm()
    D = g.delta()
    exact = netcfr(g, "cfr+", T=1000, log_every=50, track_last=False)
    quant = []
    for q in (1, 2, 3, 4, 6, 8):
        r = netcfr(g, "cfr+", T=1000, log_every=50, track_last=False, channel=quantizer(q))
        eps = 1.0 / (2 * (2**q - 1))
        quant.append(dict(q=q, bits_per_iter=g.bits_per_iteration(q),
                          nashconv_per_agent=g.nashconv(*r["avg"])[0] / g.n,
                          curve=dict(t=r["log"].t, nc_per_agent=[x / g.n for x in r["log"].nc_avg]),
                          bound_term_per_agent=4 * eps * D.sum() / g.n))
    noise = []
    for s in (0.01, 0.03, 0.1, 0.3):
        r = netcfr(g, "cfr+", T=1000, log_every=50, track_last=False, channel=laplace(s, seed=23))
        noise.append(dict(sigma=s, nashconv_per_agent=g.nashconv(*r["avg"])[0] / g.n,
                          curve=dict(t=r["log"].t, nc_per_agent=[x / g.n for x in r["log"].nc_avg])))
    lp = solve_ne(g)
    bits_edge_id = 2 * int(np.ceil(np.log2(g.n)))
    save("communication", dict(
        instance=g.summary(), sum_delta=float(D.sum()),
        exact=dict(nashconv_per_agent=g.nashconv(*exact["avg"])[0] / g.n,
                   curve=dict(t=exact["log"].t, nc_per_agent=[x / g.n for x in exact["log"].nc_avg])),
        quantized=quant, laplace=noise,
        centralized=dict(bits_once=g.E * (bits_edge_id + 2 * 32), lp_seconds=lp["seconds"],
                         lp_nashconv_per_agent=g.nashconv(lp["beta"], lp["gamma"])[0] / g.n)))


def topology():
    rows = []
    for fam in ("ba", "sbm", "er", "ws"):
        for r in (0, 1, 2):
            g = ins.topology_instance(fam, r)
            res = netcfr(g, "cfr+", T=5000, log_every=0, track_last=False,
                         stop_nc_per_agent=1e-3, check_every=10)
            rows.append(dict(family=fam, r=r, n=g.n, edges=g.E, iterations=res["stopped_at"]))
            print("   ", rows[-1], flush=True)
    save("topology", dict(rows=rows))


SNAPS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000]


def network():
    """Per-agent view: layout, equilibrium payoffs, security values, and exploitability gaps of
    each learner over time."""
    out = {}
    for name, maker, learners in (("er40", ins.er40, [("cfr+", 1), ("pcfr+", 1), ("omwu", 10), ("cfr", 1)]),
                                  ("ba100", ins.ba100, [("cfr+", 1), ("pcfr+", 1), ("omwu", 50), ("cfr", 1)])):
        g = maker()
        G = nx.Graph(); G.add_nodes_from(range(g.n)); G.add_edges_from(g.edges)
        pos = nx.kamada_kawai_layout(G)
        xy = np.array([pos[i] for i in range(g.n)])
        xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
        lp = solve_ne(g)
        _, u_eq, _ = g.nashconv(lp["beta"], lp["gamma"])
        sec = [security_value(g, i) for i in range(g.n)]
        runs = {}
        for v, eta in learners:
            r = netcfr(g, v, T=3000, eta=eta, log_every=0, snapshots=SNAPS)
            runs[v + (f"_eta{eta}" if v in ("hedge", "omwu") else "")] = r["snapshots"]
        out[name] = dict(
            n=g.n, K=g.M,
            nodes=[dict(id=i, x=float(xy[i, 0]), y=float(xy[i, 1]), degree=int(g.deg[i]),
                        eq_payoff=float(u_eq[i]), security=float(sec[i]),
                        ratios=sorted({float(g.bet_edges[e]) for e, (a, b) in enumerate(g.edges) if i in (a, b)}))
                   for i in range(g.n)],
            edges=[dict(s=int(a), t=int(b), ratio=float(g.bet_edges[e])) for e, (a, b) in enumerate(g.edges)],
            runs=runs)
        print("   ", name, "done", flush=True)
    save("network", out)


EXPS = dict(ba100=ba100_learners, er40=er40_learners, scaling=scaling, communication=communication,
            topology=topology, network=network)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(EXPS))
    a = ap.parse_args()
    for k in a.only or EXPS:
        run(k, EXPS[k])
    (RES / "run_info.json").write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%d"), python=platform.python_version(),
        machine=platform.processor() or platform.machine()), indent=1))
