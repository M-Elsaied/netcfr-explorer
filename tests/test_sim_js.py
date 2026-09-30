"""Parity of the in-browser engine (site/sim.js) with the Python implementation."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from netcfr import instances as ins
from netcfr.solvers import error_feedback_quantizer, netcfr, quantizer, stochastic_quantizer

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

SCRIPT = """
const S = require(process.argv[1]);
const cfg = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const g = new S.Game(cfg.spec);
const out = {};
for (const job of cfg.jobs) {
  const r = new S.Run(g, job);
  for (let t = 0; t < cfg.T; t++) r.step();
  out[job.key] = r.evaluate() / g.n;
}
process.stdout.write(JSON.stringify(out));
"""


def spec_of(g):
    return dict(n=g.n, M=g.M, edges=[[a, b, float(g.bet_edges[e]), float(g.stake_edges[e])]
                                     for e, (a, b) in enumerate(g.edges)])


def run_js(g, jobs, T):
    cfg = json.dumps(dict(spec=spec_of(g), jobs=jobs, T=T))
    res = subprocess.run([NODE, "-e", SCRIPT, str(ROOT / "site" / "sim.js")], input=cfg,
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def run_py(g, variant, channel, T):
    r = netcfr(g, variant, T=T, log_every=0, track_last=False, channel=channel)
    return g.nashconv(*r["avg"])[0] / g.n


@pytest.mark.parametrize("variant", ["cfr+", "cfr"])
def test_exact_dynamics_match_python(variant):
    """Without quantization the dynamics are continuous, so the engines must agree closely."""
    g = ins.er100_comm()
    js = run_js(g, [dict(key="exact", variant=variant, channel="exact")], 400)
    assert js["exact"] == pytest.approx(run_py(g, variant, None, 400), rel=1e-6)


QUANT = """
const S = require(process.argv[1]);
const cfg = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const g = new S.Game(cfg.spec);
const out = {};
for (const [key, channel, q] of cfg.cases) {
  const r = new S.Run(g, {channel, q});
  const res = [];
  for (const row of cfg.inputs) {                      // successive messages through one channel
    res.push(row.map((x, k) => r.quantize(x, k, r.eb)));
  }
  out[key] = res;
}
process.stdout.write(JSON.stringify(out));
"""


def test_rounding_functions_match_numpy_bitwise():
    """Deterministic and error-feedback rounding, including exact half-way ties."""
    g = ins.er100_comm()
    rng = np.random.default_rng(3)
    base = rng.random((6, 40))
    base[:, :8] = [0.5, 1 / 6, 0.25, 0.75, 1 / 14, 0.0, 1.0, 5 / 6]    # tie cases for L = 1, 3, 7
    cases = [("det1", "det", 1), ("det2", "det", 2), ("det3", "det", 3), ("ef1", "ef", 1), ("ef2", "ef", 2)]
    cfg = json.dumps(dict(spec=spec_of(g), inputs=base.tolist(), cases=cases))
    js = json.loads(subprocess.run([NODE, "-e", QUANT, str(ROOT / "site" / "sim.js")], input=cfg,
                                   capture_output=True, text=True, check=True).stdout)
    for key, kind, q in cases:
        L = 2**q - 1
        e = np.zeros(base.shape[1])
        for row, got in zip(base, js[key]):
            if kind == "det":
                want = np.round(row * L) / L
            else:
                v = row + e
                want = np.clip(np.round(v * L) / L, 0, 1)
                e = v - want
            assert np.array_equal(np.array(got), want), (key, row[:8], got[:8], want[:8])


@pytest.mark.parametrize("variant", ["cfr+", "cfr"])
def test_quantized_dynamics_agree_with_python(variant):
    """Rounding is discontinuous, so last-bit floating-point differences can flip individual
    roundings and the trajectories separate slightly; the outcome must still agree closely."""
    g = ins.er100_comm()
    T = 400
    cases = [("det1", ("det", 1)), ("det2", ("det", 2)), ("ef1", ("ef", 1)), ("ef3", ("ef", 3))]
    js = run_js(g, [dict(key=k, variant=variant, channel=c, q=q) for k, (c, q) in cases], T)
    for k, (c, q) in cases:
        ch = quantizer(q) if c == "det" else error_feedback_quantizer(q)
        assert js[k] == pytest.approx(run_py(g, variant, ch, T), rel=0.15), (variant, k)


def test_stochastic_channel_matches_python_in_distribution():
    g = ins.er100_comm()
    T = 400
    js = run_js(g, [dict(key=f"s{s}", variant="cfr+", channel="sto", q=1, seed=s) for s in range(6)], T)
    py = [run_py(g, "cfr+", stochastic_quantizer(1, seed=s), T) for s in range(6)]
    assert np.mean(list(js.values())) == pytest.approx(np.mean(py), rel=0.05)
