import asyncio
import logging
import random
from collections import defaultdict, deque
import heapq
import math
from typing import Optional, List, Dict, Set, Tuple, Any

import torch
import torch.functional as F
import torch.nn as nn

from .config import ModelConfig
from .core import LinearLayer

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
    def __init__(self):
        self.concepts = {}
        self.connections = {}
        self._module = None
        self._time = 0
        self.decay_rate = 0.1

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        # Remove the module since it's not picklable
        state['_module'] = None
        return state

    def __setstate__(self, state):
        """Set state during unpickling"""
        self.__dict__.update(state)
        # Reinitialize any non-picklable attributes
        self._module = None

    def add_concept(self, name: str, metadata: Optional[Dict] = None) -> str:
        """Add a concept to the graph"""
        if name not in self.concepts:
            self.concepts[name] = Concept(name, metadata)
            self.connections[name] = []
        return name

    def add_relationship(self, source: str, target: str, metadata: Optional[Dict] = None) -> None:
        """Add a relationship between concepts"""
        if source not in self.concepts or target not in self.concepts:
            raise ValueError("Both source and target concepts must exist")
        
        connection = Connection(target, metadata=metadata)
        self.connections[source].append(connection)

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name"""
        return self.concepts.get(name)

    def get_connections(self, concept: str) -> List[Connection]:
        """Get all connections from a concept"""
        return self.connections.get(concept, [])

    def remove_concept(self, name: str) -> None:
        """Remove a concept and its connections"""
        if name in self.concepts:
            del self.concepts[name]
            del self.connections[name]
            # Remove connections to this concept
            for connections in self.connections.values():
                connections[:] = [c for c in connections if c.target != name]

    def update_concept(self, name: str, metadata: Dict) -> None:
        """Update concept metadata"""
        if name in self.concepts:
            self.concepts[name].metadata.update(metadata)

    def update_relationship(self, source: str, target: str, metadata: Dict) -> None:
        """Update relationship metadata"""
        if source in self.connections:
            for conn in self.connections[source]:
                if conn.target == target:
                    conn.metadata.update(metadata)
                    break

    def get_related_concepts(self, concept: str, max_depth: int = 1) -> Set[str]:
        """Get related concepts up to max_depth"""
        if concept not in self.concepts:
            return set()
            
        related = set()
        queue = [(concept, 0)]
        visited = {concept}
        
        while queue:
            current, depth = queue.pop(0)
            if depth < max_depth:
                for conn in self.connections[current]:
                    if conn.target not in visited:
                        visited.add(conn.target)
                        related.add(conn.target)
                        queue.append((conn.target, depth + 1))
                        
        return related

    def get_concept_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between concepts"""
        if start not in self.concepts or end not in self.concepts:
            return None
            
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            for conn in self.connections[current]:
                if conn.target == end:
                    return path + [end]
                if conn.target not in visited:
                    visited.add(conn.target)
                    queue.append((conn.target, path + [conn.target]))
                    
        return None

    def get_concept_subgraph(self, concepts: Set[str]) -> "ConceptGraph":
        """Extract subgraph containing only specified concepts"""
        subgraph = ConceptGraph()
        
        # Copy concepts
        for name in concepts:
            if name in self.concepts:
                subgraph.concepts[name] = self.concepts[name]
                
        # Copy relevant connections
        for name in concepts:
            if name in self.connections:
                subgraph.connections[name] = [
                    conn for conn in self.connections[name]
                    if conn.target in concepts
                ]
                
        return subgraph

    def merge_graphs(self, other: "ConceptGraph") -> None:
        """Merge another graph into this one"""
        # Merge concepts
        for name, concept in other.concepts.items():
            if name not in self.concepts:
                self.concepts[name] = concept
            else:
                # Update existing concept
                self.concepts[name].metadata.update(concept.metadata)
                
        # Merge connections
        for name, connections in other.connections.items():
            if name not in self.connections:
                self.connections[name] = connections
            else:
                # Add new connections
                existing_targets = {conn.target for conn in self.connections[name]}
                for conn in connections:
                    if conn.target not in existing_targets:
                        self.connections[name].append(conn)

    def compute_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Compute similarity between two concepts"""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
            
        # Get sets of related concepts
        related1 = self.get_related_concepts(concept1)
        related2 = self.get_related_concepts(concept2)
        
        # Compute Jaccard similarity
        intersection = len(related1.intersection(related2))
        union = len(related1.union(related2))
        
        return intersection / union if union > 0 else 0.0

    def get_concept_clusters(self, threshold: float = 0.5) -> List[Set[str]]:
        """Cluster concepts based on similarity"""
        clusters = []
        unclustered = set(self.concepts.keys())
        
        while unclustered:
            # Start new cluster with first unclustered concept
            current = unclustered.pop()
            cluster = {current}
            
            # Find similar concepts
            for concept in list(unclustered):
                if self.compute_concept_similarity(current, concept) >= threshold:
                    cluster.add(concept)
                    unclustered.remove(concept)
                    
            clusters.append(cluster)
            
        return clusters

    def prune_weak_connections(self, threshold: float = 0.1) -> None:
        """Remove connections with low weights"""
        for name in self.connections:
            self.connections[name] = [
                conn for conn in self.connections[name]
                if conn.weight >= threshold
            ]

    def get_central_concepts(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most central concepts based on connection count"""
        centrality = {}
        for name in self.concepts:
            # Outgoing connections
            out_degree = len(self.connections[name])
            # Incoming connections
            in_degree = sum(
                1 for conns in self.connections.values()
                for conn in conns if conn.target == name
            )
            centrality[name] = out_degree + in_degree
            
        return sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

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

    def tag_concept(self, name: str, tag: str) -> None:
        if name in self.concepts:
            self.concepts[name].tags.add(tag)

    def describe_concept(self, name: str, description: str) -> None:
        if name in self.concepts:
            self.concepts[name].description = description

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

    def query_by_relationship(self, rel: str) -> List[Tuple[str,str]]:
        return [
            (c.name, conn.target)
            for c in self.concepts.values()
            for conn in c.connections
            if conn.relationship == rel
        ]

    def add_node(self, name: str, energy: float = 0.0) -> Concept:
        if name in self.concepts:
            raise ValueError(f"Node '{name}' already exists.")
        concept = Concept(name=name, energy=energy)
        self.concepts[name] = concept
        return concept

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> Connection:
        if source not in self.concepts or target not in self.concepts:
            raise ValueError("Both source and target nodes must exist.")
        connection = Connection(target=target, weight=weight)
        self.connections[source].append(connection)
        return connection

    def get_node(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)

    def get_edge(self, source: str, target: str) -> Optional[Connection]:
        if source in self.connections:
            for connection in self.connections[source]:
                if connection.target == target:
                    return connection
        return None

class ConceptSystem(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.concept_graph = ConceptGraph()
        self.concept_projector = LinearLayer(
            self.config.d_model,
            self.config.concept_dim,
            factorized=self.config.factorized_linear,
            kronecker_rank=self._valid_kronecker_rank(
                self.config.kronecker_rank,
                self.config.d_model,
                self.config.concept_dim
            )
        )
        
    def get_concept_projections(self, x):
        concept_feats = self.concept_projector(x)
        norms = torch.norm(concept_feats, dim=-1, keepdim=True)
        return concept_feats / (norms + 1e-8)