"""Graph generators and bet assignment with fixed, documented seed rules, so every benchmark
instance is regenerated exactly:

* make_graph: NetworkX seed = default_rng(seed).integers(0, 2**31 - 1)
* assign_bets: stream default_rng(seed + 7919), one draw per edge in NetworkX edge order
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .game import NSP


def nx_seed(seed: int) -> int:
    return int(np.random.default_rng(seed).integers(0, 2**31 - 1))


def make_graph(kind: str, n: int, seed: int = 0, **kw) -> nx.Graph:
    s = nx_seed(seed)
    if kind == "er":
        d = kw.get("d", 4)
        return nx.gnp_random_graph(n, min(1.0, d / (n - 1)), seed=s)
    if kind == "ws":
        return nx.connected_watts_strogatz_graph(n, kw.get("k", 4), kw.get("p", 0.1), seed=s)
    if kind == "ba":
        return nx.barabasi_albert_graph(n, kw.get("m", 2), seed=s)
    if kind == "sbm":
        blocks = kw.get("blocks", 4)
        pin, pout = kw.get("pin", 0.2), kw.get("pout", 0.01)
        sizes = [n // blocks] * blocks
        sizes[-1] += n - sum(sizes)
        probs = [[pin if a == b else pout for b in range(blocks)] for a in range(blocks)]
        return nx.stochastic_block_model(sizes, probs, seed=s)
    raise ValueError(f"unknown graph kind {kind!r}")


def assign_bets(G: nx.Graph, scheme: str = "hom", seed: int = 0, values=(1.0, 3.0)) -> list[float]:
    """One bet ratio per edge, in NetworkX edge order."""
    if scheme == "hom":
        return [float(values[0])] * G.number_of_edges()
    if scheme == "mix":
        rng = np.random.default_rng(seed + 7919)
        return [float(rng.choice(values)) for _ in G.edges()]
    raise ValueError(f"unknown bet scheme {scheme!r}")


def build(G: nx.Graph, bets: list[float], K: int = 3, stakes: list[float] | None = None) -> NSP:
    edges = list(G.edges())
    w = stakes if stakes is not None else [1.0] * len(edges)
    return NSP(G.number_of_nodes(), edges, bets, w, K)


# ---- benchmark instances ---------------------------------------------------------------

def er40() -> NSP:
    """ER-40: ER, 40 players, d=4, mixed ratios {1,3}, K=3."""
    G = make_graph("er", 40, seed=909, d=4)
    return build(G, assign_bets(G, "mix", seed=9), K=3)


def ba100() -> NSP:
    """BA-100: BA, 100 players, m=2, ratios {0.5,1,2,4}, K=5."""
    G = make_graph("ba", 100, seed=4242, m=2)
    return build(G, assign_bets(G, "mix", seed=0, values=(0.5, 1.0, 2.0, 4.0)), K=5)


def er_scaling(n: int) -> NSP:
    """Scalability instances: fast_gnp_random_graph(n, 6/(n-1), seed=7), ratios {1,3} seed 1, K=5."""
    G = nx.fast_gnp_random_graph(n, 6 / (n - 1), seed=7)
    return build(G, assign_bets(G, "mix", seed=1), K=5)


def er100_comm() -> NSP:
    """Quantized-communication instance: ER n=100, seed 2300, d=4, ratios {1,3} seed 23, K=3."""
    G = make_graph("er", 100, seed=2300, d=4)
    return build(G, assign_bets(G, "mix", seed=23), K=3)


TOPOLOGY_FAMILIES = {
    "er": dict(kind="er", d=6),
    "ws": dict(kind="ws", k=6, p=0.1),
    "ba": dict(kind="ba", m=3),
    "sbm": dict(kind="sbm", blocks=4, pin=0.2, pout=0.01),
}


def topology_instance(family: str, r: int, n: int = 100, K: int = 3) -> NSP:
    """Topology sensitivity: graph seed 990 + r, bet seed r ."""
    spec = dict(TOPOLOGY_FAMILIES[family])
    kind = spec.pop("kind")
    G = make_graph(kind, n, seed=990 + r, **spec)
    return build(G, assign_bets(G, "mix", seed=r), K=K)
