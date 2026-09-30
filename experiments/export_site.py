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
