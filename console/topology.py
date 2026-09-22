"""The dependency graph as the console sees it.

SABLE's engine reasons over node INDICES and its server maps them to identity
positionally against the topology YAML (docker/server.py:86-106); it computes
no adjacency at all (INTEGRATION-NOTES B5). So the console owns the graph:
this module loads the same edges — from `GET /api/topology` or the YAML —
and answers the two questions the loop needs:

  dependents(node)    who breaks when `node` breaks (downstream impact)
  dependencies(node)  what `node` needs (upstream causes)

Edge convention, from sable_sim/core/graph.py:59-70: an edge source→target
means "source depends on target". So the dependents of X are the sources of
edges whose target is X — transitively.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


class Topology:
    def __init__(self, nodes: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]], name: str = ""):
        self.name = name
        self.nodes: dict[str, dict[str, Any]] = {}
        for i, n in enumerate(nodes):
            nid = str(n["id"])
            self.nodes[nid] = {**n, "id": nid, "index": i}   # index = position = SABLE's node index
        self.edges: list[dict[str, Any]] = []
        self._needs: dict[str, set[str]] = {}      # source -> targets it depends on
        self._needed_by: dict[str, set[str]] = {}  # target -> sources that depend on it
        for e in edges:
            s, t = str(e["source"]), str(e["target"])
            self.edges.append({**e, "source": s, "target": t})
            self._needs.setdefault(s, set()).add(t)
            self._needed_by.setdefault(t, set()).add(s)

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Topology":
        """The dict `GET /api/topology` returns: {name, nodes[], edges[]}."""
        return cls(payload.get("nodes") or [], payload.get("edges") or [], payload.get("name", ""))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Topology":
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        return cls(data.get("nodes") or [], data.get("edges") or [], p.stem)

    # ---------------------------------------------------------------- identity

    def index_of(self, node_id: str) -> Optional[int]:
        n = self.nodes.get(node_id)
        return None if n is None else n["index"]

    def id_at(self, index: int) -> Optional[str]:
        for nid, n in self.nodes.items():
            if n["index"] == index:
                return nid
        return None

    def component_type(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("type", "UNKNOWN"))

    def label(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("label", node_id))

    # ---------------------------------------------------------------- reachability

    def _walk(self, start: str, adj: dict[str, set[str]], hard_only: bool = False) -> list[str]:
        seen, order, q = {start}, [], deque([start])
        while q:
            cur = q.popleft()
            for nxt in sorted(adj.get(cur, ())):
                if hard_only and not self._edge_is_hard(cur, nxt, adj is self._needs):
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    order.append(nxt)
                    q.append(nxt)
        return order

    def _edge_is_hard(self, a: str, b: str, a_is_source: bool) -> bool:
        s, t = (a, b) if a_is_source else (b, a)
        for e in self.edges:
            if e["source"] == s and e["target"] == t:
                return str(e.get("criticality", "SOFT")).upper() == "HARD"
        return False

    def dependents(self, node_id: str, hard_only: bool = False) -> list[str]:
        """Transitive downstream impact: everything that (directly or through
        others) depends on node_id. BFS order, node_id excluded."""
        return self._walk(node_id, self._needed_by, hard_only)

    def dependencies(self, node_id: str, hard_only: bool = False) -> list[str]:
        """Transitive upstream needs: everything node_id depends on."""
        return self._walk(node_id, self._needs, hard_only)

    def blast_radius(self, node_id: str) -> list[str]:
        """The plan's blast radius: the target plus its downstream dependents."""
        return [node_id] + [n for n in self.dependents(node_id) if n != node_id]

    def direct_dependents(self, node_id: str) -> list[str]:
        return sorted(self._needed_by.get(node_id, ()))

    def direct_dependencies(self, node_id: str) -> list[str]:
        return sorted(self._needs.get(node_id, ()))
