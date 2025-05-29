from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Optional, Dict, Set, List, Tuple, Any
import random, logging, copy
import torch

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class Connection:
    target: str
    weight: float = 1.0
    relationship: str = "co-occurrence"
    confidence: float = 1.0

@dataclass(slots=True)
class Concept:
    name: str
    energy: float = 0.0
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    domain: Optional[str] = None
    source: Optional[str] = None
    token_ids: Set[int] = field(default_factory=set)
    embedding: Optional[torch.Tensor] = None
    access_count: int = 0
    last_access_time: int = 0
    regions: Set[str] = field(default_factory=set)

class ConceptGraph:
    __slots__ = (
        "concepts", "adj", "time", "decay_rate",
        "_in_degree", "_out_degree",
        "regions", "region_meta"
    )
    
    def __init__(self, decay_rate: float = 0.1):
        self.concepts: Dict[str, Concept] = {}
        self.adj: Dict[str, List[Connection]] = defaultdict(list)
        self.time = 0
        self.decay_rate = decay_rate
        self._in_degree: Dict[str,int] = defaultdict(int)
        self._out_degree: Dict[str,int] = defaultdict(int)
        self.regions: Dict[str, Set[str]] = {}            # region_name → set of concepts
        self.region_meta: Dict[str, Any] = {}             # region_name → purpose/metadata

    def add_concept(self, name: str, **meta) -> Concept:
        if name not in self.concepts:
            self.concepts[name] = Concept(name, **meta)
        return self.concepts[name]

    def remove_concept(self, name: str):
        if name not in self.concepts:
            return
        # remove from regions
        for r in list(self.concepts[name].regions):
            self.regions.get(r, set()).discard(name)
        # delete concept & adjust degrees
        del self.concepts[name]
        out_conns = self.adj.pop(name, [])
        for c in out_conns:
            self._in_degree[c.target] -= 1
        for src, conns in self.adj.items():
            keep = []
            for c in conns:
                if c.target == name:
                    self._out_degree[src] -= 1
                    self._in_degree[name] -= 1
                else:
                    keep.append(c)
            self.adj[src] = keep

    def add_edge(self, src: str, tgt: str, **kw) -> Connection:
        if src not in self.concepts or tgt not in self.concepts:
            raise KeyError("Both source and target concepts must exist")
        conn = Connection(target=tgt, **kw)
        self.adj[src].append(conn)
        self._out_degree[src] += 1
        self._in_degree[tgt] += 1
        return conn

    def get(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)

    def neighbors(self, name: str) -> List[Connection]:
        return self.adj.get(name, [])

    def propagate_energy(self):
        decay = self.decay_rate
        new_e = {n: 0.0 for n in self.concepts}
        for n, concept in self.concepts.items():
            e = concept.energy
            if e <= 0.0:
                continue
            retained = e * decay
            new_e[n] += retained
            prop_total = e - retained
            conns = self.adj.get(n)
            if not conns:
                new_e[n] += prop_total
            else:
                wsum = sum(c.weight for c in conns)
                if wsum > 0:
                    factor = prop_total / wsum
                    for c in conns:
                        new_e[c.target] += c.weight * factor
        for n, e in new_e.items():
            self.concepts[n].energy = e
        logger.debug("Energy propagated")

    def decay(self):
        r = self.decay_rate
        for c in self.concepts.values():
            c.energy *= r
        self.time += 1
        logger.debug("Global decay applied, time=%d", self.time)
        return self.time

    def find_path(self, start: str, end: str, max_hops: int = 4) -> List[str]:
        if start not in self.concepts or end not in self.concepts:
            return []
        q = deque([[start]])
        seen = {start}
        while q:
            path = q.popleft()
            if len(path) > max_hops:
                continue
            last = path[-1]
            for c in self.adj.get(last, ()):
                tgt = c.target
                if tgt in seen:
                    continue
                newp = path + [tgt]
                if tgt == end:
                    return newp
                seen.add(tgt)
                q.append(newp)
        return []

    def sample_path(self, start: str, max_len: int = 6) -> List[str]:
        if start not in self.concepts:
            return []
        path = [start]
        for _ in range(max_len):
            conns = self.adj.get(path[-1], ())
            if not conns:
                break
            weights = [(self.concepts[c.target].energy + 1e-6) * c.weight for c in conns]
            chosen = random.choices(conns, weights)[0]
            path.append(chosen.target)
        return path

    def similarity(self, a: str, b: str) -> float:
        if a not in self.concepts or b not in self.concepts:
            return 0.0
        sa = {c.target for c in self.adj.get(a, ())}
        sb = {c.target for c in self.adj.get(b, ())}
        inter = sa & sb
        uni = sa | sb
        return len(inter) / len(uni) if uni else 0.0

    def centrality(self, k: int = 5) -> List[Tuple[str, int]]:
        scores = {n: self._in_degree.get(n, 0) + self._out_degree.get(n, 0)
                  for n in self.concepts}
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def prune(self, energy_thresh: float, access_thresh: int = 0):
        to_del = [n for n, c in self.concepts.items()
                  if c.energy < energy_thresh and c.access_count < access_thresh]
        for n in to_del:
            self.remove_concept(n)
            logger.debug("Pruned concept %s", n)

    def cluster_concepts(self, threshold: float) -> List[Set[str]]:
        names = list(self.concepts)
        clusters: List[Set[str]] = []
        assigned: Set[str] = set()
        for name in names:
            if name in assigned:
                continue
            cluster = {name}
            assigned.add(name)
            for other in names:
                if other not in assigned and self.similarity(name, other) >= threshold:
                    cluster.add(other)
                    assigned.add(other)
            clusters.append(cluster)
        logger.debug("Formed %d clusters (thresh=%.2f)", len(clusters), threshold)
        return clusters

    def associate_token(
        self,
        concept_name: str,
        token_id: int,
        embedding: Optional[torch.Tensor] = None,
        alpha: float = 0.9
    ):
        """Bind a token (and optional embedding) to a concept."""
        c = self.add_concept(concept_name)
        c.token_ids.add(token_id)
        if embedding is not None:
            emb = embedding.detach()
            if c.embedding is None:
                c.embedding = emb.clone()
            else:
                c.embedding = alpha * c.embedding + (1 - alpha) * emb
        logger.debug("Associated token %d → %s", token_id, concept_name)

    def get_concepts_by_token(self, token_id: int) -> List[str]:
        return [n for n, c in self.concepts.items() if token_id in c.token_ids]

    def define_region(self, region_name: str, purpose: Any = None):
        if region_name not in self.regions:
            self.regions[region_name] = set()
            self.region_meta[region_name] = purpose
        logger.debug("Defined region %s (purpose=%r)", region_name, purpose)

    def add_to_region(self, region_name: str, concept_name: str):
        if concept_name not in self.concepts:
            raise KeyError(f"Concept {concept_name!r} not found")
        self.define_region(region_name)
        self.regions[region_name].add(concept_name)
        self.concepts[concept_name].regions.add(region_name)
        logger.debug("Added %s to region %s", concept_name, region_name)

    def get_region(self, region_name: str) -> Optional[Set[str]]:
        return self.regions.get(region_name)

    def get_concept_subgraph(self, names: Set[str]) -> ConceptGraph:
        """Extract a new graph over a subset of concepts."""
        sub = ConceptGraph(decay_rate=self.decay_rate)
        # copy concepts
        for n in names:
            if n in self.concepts:
                sub.concepts[n] = copy.deepcopy(self.concepts[n])
        # copy edges among them
        for src in names:
            for c in self.adj.get(src, ()):
                if c.target in names:
                    sub.adj[src].append(copy.deepcopy(c))
                    sub._out_degree[src] += 1
                    sub._in_degree[c.target] += 1
        return sub

    def clear(self):
        self.concepts.clear()
        self.adj.clear()
        self._in_degree.clear()
        self._out_degree.clear()
        self.regions.clear()
        self.region_meta.clear()
        self.time = 0
        logger.debug("Cleared entire graph")