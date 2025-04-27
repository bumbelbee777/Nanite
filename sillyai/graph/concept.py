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

    def add_connection(self, target: str, weight: float, relationship: str) -> None:
        self.connections.append(Connection(target, weight, relationship))


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
    def __init__(self, decay_rate=0.9):
        self.concepts = {}
        self.decay_rate = decay_rate
        self._time = 0

    def add_concept(self, name):
        if name not in self.concepts:
            self.concepts[name] = Concept(name)

    def add_connection(self, source, target, weight=1.0, relationship="co-occurrence"):
        if source not in self.concepts or target not in self.concepts:
            return

        # Add bidirectional connections
        self.concepts[source].add_connection(target, weight, relationship)
        self.concepts[target].add_connection(source, weight, relationship)

    def propagate_energy(self):
        """Propagate energy through concept graph with proper distribution."""
        # Save initial energies and prepare new energy state
        initial_energies = {name: concept.energy for name, concept in self.concepts.items()}
        new_energies = {name: 0.0 for name in self.concepts}

        # First pass: calculate energy distribution
        for name, concept in self.concepts.items():
            current_energy = initial_energies[name]
            if current_energy <= 0:
                continue

            # Calculate retained vs propagated energy
            retained = current_energy * self.decay_rate
            new_energies[name] += retained
            
            # Calculate propagation energy
            propagate_energy = current_energy * (1 - self.decay_rate)
            if not concept.connections:
                new_energies[name] += propagate_energy  # Keep remaining energy if no connections
                continue

            # Normalize connection weights for fair distribution
            total_weight = sum(conn.weight for conn in concept.connections)
            if total_weight > 0:
                for conn in concept.connections:
                    target = conn.target
                    if target in self.concepts:  # Ensure target exists
                        # Calculate energy share based on connection weight
                        energy_share = propagate_energy * (conn.weight / total_weight)
                        new_energies[target] += energy_share

        # Update all concept energies with new values
        for name, energy in new_energies.items():
            self.concepts[name].energy = energy

        logger.debug(f"Energy propagation complete. New energies: {new_energies}")

    def update_n_cluster(self, min_energy=0.1, purge_threshold=10):
        """Update concept cluster, removing low energy/access concepts."""
        to_remove = []
        for name, concept in self.concepts.items():
            if concept.energy < min_energy and concept.access_count < purge_threshold:
                to_remove.append(name)

        for name in to_remove:
            self._remove_concept(name)

    def _remove_concept(self, name):
        """Remove a concept and its connections."""
        if name not in self.concepts:
            return

        # Remove connections to this concept from others
        concept = self.concepts[name]
        for conn in concept.connections:
            other = self.concepts.get(conn.target)
            if other:
                other.connections = [c for c in other.connections if c.target != name]

        # Remove the concept itself
        del self.concepts[name]

    def _update_access(self, name):
        """Update access count and time for a concept."""
        if name in self.concepts:
            self.concepts[name].access_count += 1
            self.concepts[name].last_access_time = self._time
            self._time += 1

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
        return [name for name, concept in self.concepts.items() if token_id in concept.token_ids]

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
