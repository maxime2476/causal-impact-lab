"""The project causal graph for the headline relative-effect design.

Encodes the directed acyclic graph behind the interacted panel local projection:
the effect of the exposure-shock interaction (``treatment = E_i * s_t``) on the
cumulative change in log employment, with the Fed reaction function as the
aggregate backdoor and a predetermined cell trend as the cross-sectional
backdoor.

The mapping to the estimator is deliberate: the aggregate backdoors
(``agg_conditions``, ``shock``) are closed by **time fixed effects**, and the
cross-sectional backdoors (``exposure``, ``cell_trend``) by **unit fixed
effects** and lagged controls. See ``docs/methods.md`` and ``docs/analysis_plan.md``.
"""

from __future__ import annotations

import networkx as nx

#: Treatment and outcome node identifiers.
TREATMENT = "treatment"
OUTCOME = "d_employment"

#: Graph nodes mapped to a human-readable description.
NODES: dict[str, str] = {
    "agg_conditions": (
        "Aggregate conditions driving the Fed reaction function: output gap, "
        "inflation, oil/commodity shocks, fiscal shocks, global demand."
    ),
    "shock": "Identified national monetary policy shock s_t.",
    "exposure": "Predetermined cell interest-rate exposure E_i (shift-share).",
    "cell_trend": (
        "Predetermined cell-specific trend / industry composition correlated "
        "with exposure (the conditional-parallel-trends threat)."
    ),
    TREATMENT: "Exposure-shock interaction D = E_i * s_t.",
    OUTCOME: "Cumulative change in log employment, y_{i,t+h} - y_{i,t-1}.",
}

#: Directed edges (cause -> effect).
EDGES: tuple[tuple[str, str], ...] = (
    # Fed reaction function: the aggregate backdoor.
    ("agg_conditions", "shock"),
    ("agg_conditions", OUTCOME),
    ("shock", OUTCOME),  # aggregate effect of the shock (absorbed by time FE)
    # Treatment construction.
    ("shock", TREATMENT),
    ("exposure", TREATMENT),
    # Cross-sectional backdoors.
    ("exposure", OUTCOME),
    ("cell_trend", "exposure"),
    ("cell_trend", OUTCOME),
    # The causal effect of interest.
    (TREATMENT, OUTCOME),
)


def build_graph() -> nx.DiGraph:
    """Return the causal graph as a :class:`networkx.DiGraph`.

    Returns
    -------
    networkx.DiGraph
        The directed acyclic graph with described nodes and edges.

    Raises
    ------
    ValueError
        If the constructed graph is not acyclic.
    """
    graph = nx.DiGraph()
    for node, description in NODES.items():
        graph.add_node(node, description=description)
    graph.add_edges_from(EDGES)
    if not nx.is_directed_acyclic_graph(graph):
        msg = "Causal graph must be acyclic."
        raise ValueError(msg)
    return graph


def graph_gml() -> str:
    """Return the causal graph as a GML string for DoWhy.

    Returns
    -------
    str
        A GML ``graph [...]`` representation consumable by ``dowhy.CausalModel``.
    """
    node_lines = "".join(f'node[id "{node}" label "{node}"]' for node in NODES)
    edge_lines = "".join(f'edge[source "{src}" target "{dst}"]' for src, dst in EDGES)
    return f"graph[directed 1 {node_lines}{edge_lines}]"
