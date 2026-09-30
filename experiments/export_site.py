"""Bundle results/*.json into site/data/results.json for the web explorer."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "site" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def r(x, nd=6):
    if isinstance(x, float):
        return float(f"{x:.{nd}g}")
    if isinstance(x, list):
        return [r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    return x


bundle = {}
for name in ("ba100_learners", "er40_learners", "scaling", "communication", "topology", "network", "rounding", "run_info"):
    p = RES / f"{name}.json"
    if p.exists():
        bundle[name] = r(json.loads(p.read_text()), 5)
(OUT / "results.json").write_text(json.dumps(bundle, separators=(",", ":")))
print("wrote", OUT / "results.json", f"{(OUT / 'results.json').stat().st_size / 1024:.0f} KB")


# ---- graphs for the in-browser side-by-side lab (layout computed once, deterministic)
import sys  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT))
from netcfr import instances as ins  # noqa: E402


def lab_graph(g, seed=1):
    G = nx.Graph(); G.add_nodes_from(range(g.n)); G.add_edges_from(g.edges)
    giant = max(nx.connected_components(G), key=len)
    rest = sorted(set(G) - giant)
    pos = nx.kamada_kawai_layout(G.subgraph(giant))
    xy = np.zeros((g.n, 2))
    core = np.array([pos[i] for i in sorted(giant)])
    core = (core - core.min(0)) / np.maximum(core.max(0) - core.min(0), 1e-9)
    band = 0.9 if rest else 1.0                      # leave a strip at the bottom for small components
    for i, (x, y) in zip(sorted(giant), core):
        xy[i] = (x, y * band)
    for k, i in enumerate(rest):                     # isolated / small components in a row
        xy[i] = ((k + 1) / (len(rest) + 1), 1.0)
    return dict(n=g.n, M=g.M, xy=[[round(float(a), 4), round(float(b), 4)] for a, b in xy],
                edges=[[int(a), int(b), float(g.bet_edges[e]), float(g.stake_edges[e])] for e, (a, b) in enumerate(g.edges)])


lab = dict(er100=lab_graph(ins.er100_comm()), er40=lab_graph(ins.er40()))
(OUT / "lab_graphs.json").write_text(json.dumps(lab, separators=(",", ":")))
print("wrote", OUT / "lab_graphs.json")
