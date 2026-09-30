"""Static PNG figures from results/*.json at 300 dpi."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
load = lambda n: json.loads((RES / f"{n}.json").read_text())
plt.rcParams.update({"font.size": 10})


def fig2():
    R = load("e09_er40")["runs"]
    keys = [("pcfr+", "PCFR+"), ("dcfr", "DCFR"), ("cfr+", "CFR+"), ("omwu_eta10", "OMWU"),
            ("hedge_eta10", "Hedge"), ("cfr", "CFR"), ("fp", "FP")]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, [R[k]["nc_avg"][-1] for k, _ in keys], "o-", label="Average profile")
    ax.plot(x, [R[k]["nc_last"][-1] for k, _ in keys], "s-", label="Last iterate")
    ax.set_xticks(list(x), [lab for _, lab in keys])
    ax.set_yscale("log")
    ax.set_ylabel("Total NashConv on ER-40 (log scale)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_er40.png", dpi=300)


def fig3():
    rows = load("scaling")["rows"]
    n = [r["n"] for r in rows]
    s = [r["s_per_iter"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.loglog(n, s, "o-")
    for xi, yi in zip(n, s):
        ax.annotate(f"{yi*1e3:.1f} ms" if yi < 0.1 else f"{yi:.2f} s", (xi, yi), xytext=(6, 6),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Number of agents, n")
    ax.set_ylabel("Seconds per NetCFR iteration")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scaling.png", dpi=300)


def fig4():
    d = load("communication")
    q = [r["q"] for r in d["quantized"]]
    y = [r["nashconv_per_agent"] for r in d["quantized"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(q, y, "o-", label="q-bit messages")
    ax.axhline(d["exact"]["nashconv_per_agent"], ls="--", label="exact-message baseline")
    ax.set_xlabel("Bits per transmitted probability (q)")
    ax.set_ylabel("NashConv per agent after 1,000 iterations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_quantization.png", dpi=300)


if __name__ == "__main__":
    fig2(); fig3(); fig4()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
