import asyncio
import logging
import random
from collections import defaultdict, deque
import heapq
import math
from typing import Optional, List, Dict, Set, Tuple, Any
import torch
import torch.functional as F

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Connection:
    def __init__(
        self,
        target: str,
        weight: float = 1.0,
        relationship: str = "co-occurrence",
        confidence: float = 1.0
    ):
        self.target = target
        self.weight = weight
        self.relationship = relationship
        self.confidence = confidence

    def __repr__(self):
        return (
            f"Connection(target={self.target!r}, weight={self.weight:.3f}, "
            f"relationship={self.relationship!r}, confidence={self.confidence:.3f})"
        )


class Concept:
    def __init__(
        self,
        name: str,
        energy: float = 0.0,
        description: str = "",
        domain: Optional[str] = None,
        source: Optional[str] = None
    ):
        self.name = name
        self.energy = energy
        self.connections: List[Connection] = []
        self.access_count = 0
        self.last_access_time = 0

        # --- NEW FIELDS ---
        # for token/vector association
        self.token_ids: Set[int] = set()
        self.embedding: Optional[torch.Tensor] = None

        # metadata
        self.tags: Set[str] = set()
        self.description = description
        self.domain = domain
        self.source = source

        # region membership
        self.regions: Set[str] = set()

    def __repr__(self):
        conns = ', '.join(repr(c) for c in self.connections)
        tags = ','.join(self.tags)
        toks = ','.join(str(t) for t in self.token_ids)
        regs = ','.join(self.regions)
        return (
            f"Concept(name={self.name!r}, energy={self.energy:.3f}, "
            f"tags=[{tags}], tokens=[{toks}], regions=[{regs}], "
            f"access_count={self.access_count}, last_access_time={self.last_access_time}, "
            f"connections=[{conns}])"
        )


class ReasoningStep:
    def __init__(
        self,
        frm: str,
        to: str,
        relationship: str,
        justification: Optional[str] = None,
        confidence: float = 1.0
    ):
        self.frm = frm
        self.to = to
        self.relationship = relationship
        self.justification = justification
        self.confidence = confidence

    def __repr__(self):
        return (
            f"{self.frm} -[{self.relationship}/{self.confidence:.2f}]-> {self.to}"
            + (f" {{{self.justification}}}" if self.justification else "")
        )


class GraphRegion:
    def __init__(self, name: str, purpose: Optional[str] = None):
        self.name = name
        self.purpose = purpose
        self.concepts: Set[str] = set()

    def __repr__(self):
        return f"GraphRegion(name={self.name!r}, purpose={self.purpose!r}, size={len(self.concepts)})"


class ConceptGraph:
    def __init__(self, decay_rate: float = 0.45, debug: bool = False):
        self.concepts: Dict[str, Concept] = {}
        self.decay_rate = decay_rate
        self.time = 0
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # new containers
        self.regions: Dict[str, GraphRegion] = {}

    # ----------------------------
    # Legacy API (unchanged)
    # ----------------------------
    def add_concept(self, name: str) -> None:
        if name not in self.concepts:
            self.concepts[name] = Concept(name)
            logger.debug(f"Added concept: {name}")

    def add_connection(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        relationship: str = "co-occurrence",
        confidence: float = 1.0
    ) -> None:
        self.add_concept(source)
        self.add_concept(target)
        conn = Connection(target, weight, relationship, confidence)
        self.concepts[source].connections.append(conn)
        # add reciprocal
        self.concepts[target].connections.append(
            Connection(source, weight, relationship, confidence)
        )
        logger.debug(f"Connected {source} <-> {target} "
                     f"weight={weight}, relationship={relationship}, confidence={confidence}")

    async def _propagate_energy(self, transfer_factor: float = 0.1) -> None:
        new_energy = defaultdict(float)
        for concept in list(self.concepts.values()):
            for conn in concept.connections:
                transferred = concept.energy * conn.weight * transfer_factor
                new_energy[conn.target] += transferred
                new_energy[concept.name] -= transferred
        for name, delta in new_energy.items():
            if name in self.concepts:
                c = self.concepts[name]
                old = c.energy
                c.energy = max(0.0, old + delta)
                self._update_access(name)
                logger.debug(f"Energy update {name}: {old:.3f} -> {c.energy:.3f}")
                await asyncio.sleep(0)

    def propagate_energy(self, transfer_factor: float = 0.1) -> None:
        logger.info("Starting energy propagation...")
        try:
            asyncio.run(self._propagate_energy(transfer_factor))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._propagate_energy(transfer_factor))
        logger.info("Energy propagation complete.")

    async def _update_n_cluster(
        self,
        min_energy: float = 0.01,
        purge_threshold: int = 5
    ) -> None:
        to_remove = []
        for name, concept in list(self.concepts.items()):
            old_e = concept.energy
            concept.energy *= self.decay_rate
            if concept.energy < min_energy:
                to_remove.append(name)
            logger.debug(f"Decayed {name}: {old_e:.3f} -> {concept.energy:.3f}")
            await asyncio.sleep(0)

        for name in to_remove:
            self._purge_concept(name)

        heap = [(c.energy, c.access_count, c.name)
                for c in self.concepts.values()]
        heapq.heapify(heap)
        while len(self.concepts) > purge_threshold:
            e, cnt, nm = heapq.heappop(heap)
            if nm in self.concepts:
                logger.debug(f"LFU purge: {nm} (e={e:.3f}, count={cnt})")
                self._purge_concept(nm)
            await asyncio.sleep(0)

    def update_n_cluster(
        self,
        min_energy: float = 0.01,
        purge_threshold: int = 5
    ) -> None:
        logger.info("Starting cluster update...")
        try:
            asyncio.run(self._update_n_cluster(min_energy, purge_threshold))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self._update_n_cluster(min_energy, purge_threshold)
            )
        logger.info("Cluster update complete.")

    def _update_access(self, concept_name: str) -> None:
        c = self.concepts[concept_name]
        c.access_count += 1
        c.last_access_time = self.time
        self.time += 1
        logger.debug(f"Access {concept_name}: count={c.access_count}, time={c.last_access_time}")

    def _purge_concept(self, name: str) -> None:
        if name in self.concepts:
            del self.concepts[name]
            for c in self.concepts.values():
                c.connections = [x for x in c.connections if x.target != name]
            logger.debug(f"Purged concept: {name}")

    def __str__(self) -> str:
        lines = []
        for name, c in self.concepts.items():
            conns = ', '.join(f"{x.target}({x.weight:.2f},{x.relationship})"
                              for x in c.connections)
            lines.append(f"{name}: E={c.energy:.2f}, A={c.access_count}, Conns=[{conns}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


    # ----------------------------
    # NEW API
    # ----------------------------

    # --- 1. Token/Vector Association ---
    def associate_token(
        self,
        concept_name: str,
        token_id: int,
        embedding: Optional[torch.Tensor] = None,
        alpha: float = 0.9
    ) -> None:
        """Bind a token ID (and its embedding) to a concept."""
        if concept_name not in self.concepts:
            self.add_concept(concept_name)
        c = self.concepts[concept_name]
        c.token_ids.add(token_id)
        if embedding is not None:
            if c.embedding is None:
                c.embedding = embedding.detach().clone()
            else:
                c.embedding = alpha * c.embedding + (1 - alpha) * embedding.detach()

    def get_concepts_by_token(self, token_id: int) -> List[str]:
        return [n for n, c in self.concepts.items() if token_id in c.token_ids]

    # --- 2. Tags & Metadata ---
    def tag_concept(self, name: str, tag: str) -> None:
        if name in self.concepts:
            self.concepts[name].tags.add(tag)

    def describe_concept(self, name: str, description: str) -> None:
        if name in self.concepts:
            self.concepts[name].description = description

    # --- 3. Regions/Subgraphs ---
    def define_region(self, region_name: str, purpose: Optional[str] = None) -> None:
        if region_name not in self.regions:
            self.regions[region_name] = GraphRegion(region_name, purpose)

    def add_to_region(self, region_name: str, concept_name: str) -> None:
        self.define_region(region_name)
        self.add_concept(concept_name)
        self.regions[region_name].concepts.add(concept_name)
        self.concepts[concept_name].regions.add(region_name)

    def get_region(self, region_name: str) -> Optional[GraphRegion]:
        return self.regions.get(region_name)

    # --- 4. Inference Chains & Pathfinding ---
    def find_paths(
        self,
        start: str,
        goal: str,
        max_hops: int = 4,
        method: str = "bfs"
    ) -> List[List[ReasoningStep]]:
        """Return all simple paths (up to max_hops) or BFS chains."""
        if start not in self.concepts or goal not in self.concepts:
            return []

        paths = []
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            if len(path) > max_hops:
                continue
            last = path[-1]
            if last == goal and len(path) > 1:
                # build ReasoningStep list
                steps = []
                for a, b in zip(path, path[1:]):
                    # find first connection
                    conn = next((c for c in self.concepts[a].connections if c.target == b), None)
                    if not conn:
                        break
                    steps.append(ReasoningStep(a, b, conn.relationship, confidence=conn.confidence))
                else:
                    paths.append(steps)
                continue

            for conn in self.concepts[last].connections:
                if conn.target not in path:
                    queue.append(path + [conn.target])

        return paths

    def sample_reasoning_path(
        self,
        start: str,
        goal: Optional[str] = None,
        max_len: int = 6
    ) -> List[ReasoningStep]:
        """Random-walk (Monte Carlo) from start to goal (or open-ended)."""
        if start not in self.concepts:
            return []
        steps: List[ReasoningStep] = []
        current = start
        for _ in range(max_len):
            conns = self.concepts[current].connections
            if not conns:
                break
            # sample by weighted (energy * weight)
            weights = [(self.concepts[c.target].energy + 1e-6) * c.weight for c in conns]
            idx = random.choices(range(len(conns)), weights)[0]
            conn = conns[idx]
            step = ReasoningStep(current, conn.target, conn.relationship, confidence=conn.confidence)
            steps.append(step)
            current = conn.target
            if goal is not None and current == goal:
                break
        return steps

    # --- 5. Entanglement & Shuffling ---
    def compute_entanglement(self) -> Dict[Tuple[str,str], float]:
        """Pairwise entanglement = cosine(sim) * co-activation freq."""
        ent = {}
        names = list(self.concepts.keys())
        for i, a in enumerate(names):
            for b in names[i+1:]:
                ca = self.concepts[a]
                cb = self.concepts[b]
                # rough proxy: embedding cosine if available
                sim = 0.0
                if ca.embedding is not None and cb.embedding is not None:
                    sim = F.cosine_similarity(ca.embedding, cb.embedding, dim=0).item()
                # co-activation = times accessed within window
                freq = min(ca.access_count, cb.access_count)
                ent[(a,b)] = sim * math.log1p(freq)
        return ent

    def shuffle_graph(self, key: Optional[Any] = None) -> None:
        """Reorder concept dict for locality (in-place)."""
        items = list(self.concepts.items())
        random.Random(key).shuffle(items)
        self.concepts = dict(items)

    # --- 6. Relationship Queries ---
    def query_by_relationship(self, rel: str) -> List[Tuple[str,str]]:
        return [
            (c.name, conn.target)
            for c in self.concepts.values()
            for conn in c.connections
            if conn.relationship == rel
        ]
