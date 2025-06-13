from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Optional, Dict, Set, List, Tuple, Any
import logging
import torch
import networkx as nx
import os
from datetime import datetime

from .complex_tokens import ComplexBasisSet

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
    # New fields for complex periodic function representation
    basis_weights: Optional[torch.Tensor] = None  # Weights for basis functions
    composition_weights: Dict[str, float] = field(
        default_factory=dict
    )  # Weights for composed concepts
    frequency: float = 1.0  # Base frequency for periodic function
    phase: float = 0.0  # Phase offset for periodic function


class ConceptGraph:
    """Manages a graph of concepts and their relationships."""

    def __init__(
        self, max_size: int = 1000, decay_rate: float = 0.1, num_basis: int = 32
    ):
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.num_basis = num_basis
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, Concept] = {}
        self.regions: Dict[str, Set[str]] = {}
        self.current_time = 0

        # Set up debug logger
        self.logger = logging.getLogger("concept_graph")
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join("logs", f"concept_graph_{timestamp}.log")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _log_concept_operation(
        self, operation: str, concept_name: str, details: Dict = None
    ):
        """Log concept graph operations with details."""
        msg = f"{operation}: {concept_name}"
        if details:
            msg += f" - Details: {details}"
        self.logger.debug(msg)

    def add_concept(
        self,
        name: str,
        embedding: Optional[torch.Tensor] = None,
        relationships: Optional[Dict[str, float]] = None,
        basis_weights: Optional[torch.Tensor] = None,
        composition: Optional[Dict[str, float]] = None,
        frequency: float = 1.0,
        phase: float = 0.0,
        **meta,
    ):
        """Add a concept to the graph with detailed logging."""
        if name in self.concepts:
            self._log_concept_operation(
                "Update existing concept",
                name,
                {
                    "energy": self.concepts[name].energy,
                    "new_embedding_shape": embedding.shape
                    if embedding is not None
                    else None,
                },
            )
        else:
            self._log_concept_operation(
                "Add new concept",
                name,
                {
                    "embedding_shape": embedding.shape
                    if embedding is not None
                    else None,
                    "basis_weights_shape": basis_weights.shape
                    if basis_weights is not None
                    else None,
                },
            )

        # Create or update concept
        if name not in self.concepts:
            self.concepts[name] = Concept(name=name, **meta)

        concept = self.concepts[name]
        if embedding is not None:
            concept.embedding = embedding
        if basis_weights is not None:
            concept.basis_weights = basis_weights
        if composition is not None:
            concept.composition_weights = composition
        concept.frequency = frequency
        concept.phase = phase

        # Add to graph
        self.graph.add_node(name)

        # Add relationships
        if relationships:
            for target, weight in relationships.items():
                self.graph.add_edge(name, target, weight=weight)
                self._log_concept_operation(
                    "Add relationship", name, {"target": target, "weight": weight}
                )

    def compose_concept(
        self,
        name: str,
        components: Dict[str, float],
        frequency: float = 1.0,
        phase: float = 0.0,
    ) -> Concept:
        """Compose a new concept from existing concepts with weights."""
        if not all(comp in self.graph for comp in components):
            raise ValueError("All component concepts must exist in the graph")

        # Get component concepts
        comp_concepts = [self.graph.nodes[comp]["concept"] for comp in components]

        # Combine basis weights
        basis_weights = torch.zeros(self.num_basis, dtype=torch.complex64)
        for comp, weight in zip(comp_concepts, components.values()):
            if comp.basis_weights is not None:
                basis_weights += weight * comp.basis_weights

        # Create new concept
        concept = Concept(
            name=name,
            basis_weights=basis_weights,
            composition_weights=components,
            frequency=frequency,
            phase=phase,
        )

        # Add to graph
        self.add_concept(
            name,
            basis_weights=basis_weights,
            composition=components,
            frequency=frequency,
            phase=phase,
        )

        return concept

    def get_concept_embedding(self, name: str) -> Optional[torch.Tensor]:
        """Get the embedding for a concept."""
        if name not in self.graph:
            return None

        concept = self.graph.nodes[name]["concept"]
        if concept.basis_weights is not None:
            # Generate complex periodic function representation
            x = torch.linspace(0, 1, 100)  # Sample points
            basis_funcs = self.basis_set(x)
            return torch.einsum("n,nk->k", concept.basis_weights, basis_funcs)

        return self.concepts.get(name).embedding

    def get_related_concepts(
        self, name: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k related concepts by edge weight."""
        if name not in self.graph:
            return []

        edges = self.graph.edges(name, data=True)
        related = [(target, data["weight"]) for _, target, data in edges]
        return sorted(related, key=lambda x: x[1], reverse=True)[:top_k]

    def get_top_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k concepts by PageRank centrality."""
        centrality = nx.pagerank(self.graph)
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def propagate_energy(self):
        """Propagate energy through the concept graph."""
        decay = self.decay_rate
        new_e = {n: 0.0 for n in self.graph.nodes()}
        for n in self.graph.nodes():
            e = self.graph.nodes[n]["concept"].energy
            if e <= 0.0:
                continue
            retained = e * decay
            new_e[n] += retained
            prop_total = e - retained
            edges = self.graph.edges(n, data=True)
            if not edges:
                new_e[n] += prop_total
            else:
                wsum = sum(data["weight"] for _, _, data in edges)
                if wsum > 0:
                    factor = prop_total / wsum
                    for _, target, data in edges:
                        new_e[target] += data["weight"] * factor
        for n, e in new_e.items():
            self.graph.nodes[n]["concept"].energy = e

    def decay(self):
        """Apply global decay to all concepts."""
        r = self.decay_rate
        for n in self.graph.nodes():
            self.graph.nodes[n]["concept"].energy *= r
        self.current_time += 1
        return self.current_time

    def prune(self, energy_thresh: float, access_thresh: int = 0):
        """Prune concepts below energy and access thresholds."""
        to_remove = []
        for n in self.graph.nodes():
            node_data = self.graph.nodes[n]["concept"]
            if (
                node_data.energy < energy_thresh
                and node_data.access_count < access_thresh
            ):
                to_remove.append(n)
        for n in to_remove:
            self.graph.remove_node(n)
            if n in self.concepts:
                del self.concepts[n]

    def define_region(self, region_name: str, purpose: Any = None):
        """Define a new region in the concept graph."""
        if region_name not in self.regions:
            self.regions[region_name] = set()
            self.region_meta[region_name] = purpose

    def add_to_region(self, region_name: str, concept_name: str):
        """Add a concept to a region."""
        if concept_name not in self.graph:
            raise KeyError(f"Concept {concept_name!r} not found")
        self.define_region(region_name)
        self.regions[region_name].add(concept_name)
        if "regions" not in self.graph.nodes[concept_name]["concept"].regions:
            self.graph.nodes[concept_name]["concept"].regions = set()
        self.graph.nodes[concept_name]["concept"].regions.add(region_name)

    def get_region(self, region_name: str) -> Optional[Set[str]]:
        """Get all concepts in a region."""
        return self.regions.get(region_name)

    def to_bytecode(
        self, start_concept: Optional[str] = None, max_hops: int = 5
    ) -> List[tuple]:
        """Convert concept graph to bytecode with detailed logging."""
        if not self.concepts:
            self.logger.debug("No concepts to convert to bytecode")
            return []

        self._log_concept_operation(
            "Start bytecode generation",
            "all",
            {"start_concept": start_concept, "max_hops": max_hops},
        )

        visited = set()
        bytecode = []

        def traverse(node: str, depth: int):
            if depth > max_hops or node in visited:
                return

            visited.add(node)
            concept = self.concepts[node]

            # Log concept traversal
            self._log_concept_operation(
                "Traverse concept",
                node,
                {
                    "depth": depth,
                    "energy": concept.energy,
                    "num_relationships": len(list(self.graph.edges(node))),
                },
            )

            # Add concept to bytecode
            bytecode.append(("CONCEPT", node))

            # Process relationships
            for target, data in self.graph.edges(node, data=True):
                weight = data.get("weight", 1.0)
                if weight > 0.5:  # Only process significant relationships
                    bytecode.append(("RELATE", node, target, weight))
                    traverse(target, depth + 1)

        # Start traversal
        if start_concept:
            traverse(start_concept, 0)
        else:
            # Start with highest energy concepts
            sorted_concepts = sorted(
                self.concepts.items(), key=lambda x: x[1].energy, reverse=True
            )
            for concept_name, _ in sorted_concepts:
                if concept_name not in visited:
                    traverse(concept_name, 0)

        self._log_concept_operation(
            "Complete bytecode generation",
            "all",
            {"num_instructions": len(bytecode), "num_concepts_visited": len(visited)},
        )

        return bytecode

    def update_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """Update concept embeddings."""
        for name, embedding in embeddings.items():
            if name in self.graph:
                self.graph.nodes[name]["concept"].embedding = embedding

    def get_subgraph(self, concepts: List[str]) -> nx.DiGraph:
        """Get subgraph containing only specified concepts."""
        return self.graph.subgraph(concepts).copy()

    def merge(self, other: "ConceptGraph"):
        """Merge another concept graph into this one."""
        for node in other.graph.nodes():
            if len(self.graph) < self.max_size:
                concept = other.graph.nodes[node]["concept"]
                self.add_concept(
                    name=node,
                    embedding=concept.embedding,
                    basis_weights=concept.basis_weights,
                    composition=concept.composition_weights,
                    frequency=concept.frequency,
                    phase=concept.phase,
                )

        for u, v, data in other.graph.edges(data=True):
            if u in self.graph and v in self.graph:
                self.graph.add_edge(u, v, weight=data["weight"])

    def save(self, path: str):
        """Save concept graph to file."""
        state = {
            "graph": self.graph,
            "concepts": self.concepts,
            "max_size": self.max_size,
            "decay_rate": self.decay_rate,
            "current_time": self.current_time,
            "regions": self.regions,
            "region_meta": self.region_meta,
            "num_basis": self.num_basis,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load concept graph from file."""
        state = torch.load(path)
        self.graph = state["graph"]
        self.concepts = state["concepts"]
        self.max_size = state["max_size"]
        self.decay_rate = state["decay_rate"]
        self.current_time = state["current_time"]
        self.regions = state["regions"]
        self.region_meta = state["region_meta"]
        self.num_basis = state["num_basis"]
        self.basis_set = ComplexBasisSet(self.num_basis)

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get all concept embeddings with logging."""
        embeddings = {
            name: concept.embedding
            for name, concept in self.concepts.items()
            if concept.embedding is not None
        }
        self._log_concept_operation(
            "Get embeddings",
            "all",
            {
                "num_embeddings": len(embeddings),
                "embedding_shapes": [e.shape for e in embeddings.values()],
            },
        )
        return embeddings

    def get_relationships(self) -> List[Tuple[int, int, float]]:
        """Get all relationships with logging."""
        if not self.concepts:
            self.logger.debug("No concepts in graph")
            return []

        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(self.concepts.keys())}

        # Get all edges
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            source_idx = name_to_idx[source]
            target_idx = name_to_idx[target]
            weight = data.get("weight", 1.0)
            relationships.append((source_idx, target_idx, weight))

        self._log_concept_operation(
            "Get relationships",
            "all",
            {
                "num_relationships": len(relationships),
                "unique_concepts": len(name_to_idx),
            },
        )
        return relationships
