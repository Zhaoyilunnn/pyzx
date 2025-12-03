# Canonical labeling and structure hashing for PyZX graphs
# This module provides id-independent signatures and hashes based on
# Weisfeiler–Lehman refinement and deterministic tie-breaking.

from fractions import Fraction
from typing import Any, Dict, Tuple

from .graph_s import GraphS
from ..utils import EdgeType


def _norm_phase(g: GraphS, v: int):
    p = g._phase.get(v, Fraction(1))
    try:
        return p % 2
    except Exception:
        return p


def _initial_label(
    g: GraphS, v: int, include_qubit: bool = True, include_row: bool = True
):
    base = (g.ty[v], _norm_phase(g, v))
    if include_qubit:
        base += (g._qindex.get(v, -1),)
    if include_row:
        base += (g._rindex.get(v, -1),)
    return base


def _wl_refine(g: GraphS, labels: Dict[int, Any], rounds: int = 4):
    # Weisfeiler–Lehman like refinement for a few rounds
    # https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test
    # It is a heuristic, not guaranteed to distinguish all non-isomorphic graphs
    # But if two graphs differ in WL labels, they are definitely non-isomorphic
    for _ in range(rounds):
        new_labels: Dict[int, Any] = {}
        for v in g.graph:
            nbr_multiset = sorted((labels[u], g.graph[v][u]) for u in g.graph[v])
            new_labels[v] = (labels[v], tuple(nbr_multiset))
        # Relabel to compact integers via stable sorting
        uniq: Dict[Any, int] = {}
        idx = 0
        for v, lab in sorted(new_labels.items(), key=lambda x: (x[1], x[0])):
            if lab not in uniq:
                uniq[lab] = idx
                idx += 1
        for v in g.graph:
            new_labels[v] = uniq[new_labels[v]]
        labels = new_labels
    return labels


def _signature_under_order(g: GraphS, order: Dict[int, int]):
    # order: dict old_id -> canonical index
    V = []
    for v, canon in sorted(order.items(), key=lambda x: x[1]):
        V.append(
            (g.ty[v], _norm_phase(g, v), g._qindex.get(v, -1), g._rindex.get(v, -1))
        )
    E = []
    for v in g.graph:
        for u in g.graph[v]:
            if v < u:
                a = order[v]
                b = order[u]
                if a > b:
                    a, b = b, a
                E.append((a, b, g.graph[v][u]))
    E.sort()
    return (tuple(V), tuple(E))


def canonical_signature(
    g: GraphS, include_qubit: bool = True, include_row: bool = True
) -> Tuple[Tuple[Any, ...], Tuple[Tuple[int, int, EdgeType], ...]]:
    """
    Compute a canonical, id-independent signature for the graph that is
    invariant under vertex renaming. Uses WL refinement + deterministic
    bucket ordering to produce a lexicographically minimal signature of (V, E).

    Returns a tuple (V, E) where:
      - V is a tuple of per-vertex attributes in canonical order
      - E is a tuple of edges (u, v, edge_type) with canonical indices
    """
    if not g.graph:
        return (tuple(), tuple())
    # Initial labels
    labels = {v: _initial_label(g, v, include_qubit, include_row) for v in g.graph}
    # WL refinement
    refined = _wl_refine(g, labels, rounds=4)
    # Bucket vertices by refined label and sort buckets deterministically
    lab_to_vs: Dict[Any, list] = {}
    for v, lab in refined.items():
        lab_to_vs.setdefault(lab, []).append(v)
    buckets = [
        sorted(
            lab_to_vs[k],
            key=lambda v: (g.ty[v], _norm_phase(g, v), len(g.graph[v]), v),
        )
        for k in sorted(lab_to_vs.keys())
    ]
    # Build canonical order map by concatenating buckets
    order_map: Dict[int, int] = {}
    idx = 0
    for group in buckets:
        for v in group:
            order_map[v] = idx
            idx += 1
    # Signature under this deterministic order
    return _signature_under_order(g, order_map)


def structure_hash(
    g: GraphS, include_qubit: bool = True, include_row: bool = True
) -> int:
    """
    Returns an id-independent hash depending on topology,
    vertex types/phases, and edge types.
    """
    sig = canonical_signature(g, include_qubit=include_qubit, include_row=include_row)
    return hash(sig)
