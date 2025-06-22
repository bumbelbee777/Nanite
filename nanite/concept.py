from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import torch

from .complex_tokens import ComplexBasisSet
from .constraints import LogicalConstraint, TemporalConstraint, ConstraintType
from .vm import Opcode, safe_eval

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Connection:
    """Represents a connection between concepts."""
    target: str
    weight: float = 1.0
    relationship: str = "co-occurrence"
    confidence: float = 1.0


@dataclass
class Concept:
    """Represents a concept in the graph."""
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
    basis_weights: Optional[torch.Tensor] = None
    composition_weights: Dict[str, float] = field(default_factory=dict)
    frequency: float = 1.0
    phase: float = 0.0
    
    # New fields for constraint and context handling
    constraints: List[LogicalConstraint] = field(default_factory=list)
    context_dependencies: Dict[str, float] = field(default_factory=dict)
    temporal_constraints: List[TemporalConstraint] = field(default_factory=list)
    inference_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_constraint(self, 
                      constraint: LogicalConstraint,
                      priority: float = 1.0) -> None:
        """Add a constraint to the concept."""
        constraint.priority = priority
        self.constraints.append(constraint)
        
    def add_temporal_constraint(self,
                              constraint: TemporalConstraint,
                              priority: float = 1.0) -> None:
        """Add a temporal constraint to the concept."""
        constraint.priority = priority
        self.temporal_constraints.append(constraint)
        
    def add_inference_rule(self,
                          condition: str,
                          conclusion: str,
                          confidence: float = 1.0) -> None:
        """Add an inference rule to the concept."""
        self.inference_rules.append({
            "condition": condition,
            "conclusion": conclusion,
            "confidence": confidence
        })
        
    def add_context_dependency(self,
                             context_key: str,
                             weight: float) -> None:
        """Add a context dependency to the concept."""
        self.context_dependencies[context_key] = weight
        
    def evaluate_constraints(self,
                           context: Dict[str, Any]) -> bool:
        """Evaluate all constraints in the given context."""
        # Check regular constraints
        for constraint in self.constraints:
            if not constraint.evaluate({"concept": self, "context": context}):
                return False
                
        # Check temporal constraints
        for constraint in self.temporal_constraints:
            if not constraint.evaluate({"concept": self, "context": context}):
                return False
                
        return True
        
    def apply_inference_rules(self,
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply inference rules in the given context."""
        results = []
        for rule in self.inference_rules:
            if safe_eval(rule["condition"], {"concept": self, "context": context}):
                results.append({
                    "conclusion": rule["conclusion"],
                    "confidence": rule["confidence"]
                })
        return results
        
    def get_context_weight(self,
                          context: Dict[str, Any]) -> float:
        """Calculate the weight of the concept in the given context."""
        weight = 1.0
        for key, value in self.context_dependencies.items():
            if key in context:
                weight *= value
        return weight
        
    def update_energy(self,
                     context: Dict[str, Any]) -> None:
        """Update concept energy based on context and constraints."""
        # Base energy update
        self.energy *= 0.95  # Natural decay
        
        # Add energy from context relevance
        context_weight = self.get_context_weight(context)
        self.energy += context_weight * 0.1
        
        # Add energy from constraint satisfaction
        if self.evaluate_constraints(context):
            self.energy += 0.2
            
        # Add energy from inference rule applications
        inference_results = self.apply_inference_rules(context)
        self.energy += len(inference_results) * 0.05
        
        # Cap energy at 1.0
        self.energy = min(1.0, self.energy)
        
    def to_bytecode(self,
                   context: Dict[str, Any]) -> List[Tuple[Opcode, List[str]]]:
        """Convert concept to bytecode with context awareness."""
        bytecode = []
        
        # Add concept definition
        bytecode.append((Opcode.CASSERT, [self.name]))
        
        # Add constraints as assertions
        for constraint in self.constraints:
            bytecode.append((Opcode.ASSERT, [constraint.condition]))
            
        # Add temporal constraints
        for constraint in self.temporal_constraints:
            if constraint.start_time is not None:
                bytecode.append((Opcode.ASSERT, [f"time >= {constraint.start_time}"]))
            if constraint.end_time is not None:
                bytecode.append((Opcode.ASSERT, [f"time <= {constraint.end_time}"]))
                
        # Add inference rules
        for rule in self.inference_rules:
            bytecode.append((Opcode.CIF, [rule["condition"], rule["conclusion"]]))
            
        return bytecode


class ConceptGraph:
    """Manages a graph of concepts and their relationships."""

    def __init__(
        self,
        max_size: int = 1000,
        decay_rate: float = 0.1,
        num_basis: int = 32,
    ):
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.num_basis = num_basis
        self.graph = nx.DiGraph()
        self.concepts: dict[str, Concept] = {}
        self.regions: dict[str, set[str]] = {}
        self.region_meta: dict[str, Any] = {}
        self.current_time = 0
        self.adj = {}  # Adjacency list for connections

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
        self,
        operation: str,
        concept_name: str,
        details: dict = None,
    ):
        """Log concept graph operations with details."""
        msg = f"{operation}: {concept_name}"
        if details:
            msg += f" - Details: {details}"
        self.logger.debug(msg)

    def add_edge(self, source: str, target: str, weight: float = 1.0, relationship: str = "co-occurrence"):
        """Add an edge between two concepts."""
        if source not in self.concepts or target not in self.concepts:
            raise ValueError("Both source and target concepts must exist")
            
        # Add edge to the graph
        self.graph.add_edge(source, target, weight=weight, relationship=relationship)
        
        # Add to adjacency list
        if source not in self.adj:
            self.adj[source] = []
        if target not in self.adj:
            self.adj[target] = []
            
        self.adj[source].append(Connection(target=target, weight=weight, relationship=relationship))
        self.adj[target].append(Connection(target=source, weight=weight, relationship=relationship))
        
        # Log the operation
        self._log_concept_operation(
            "Add edge",
            source,
            {"target": target, "weight": weight, "relationship": relationship}
        )

    def add_concept(
        self,
        name: str,
        embedding: torch.Tensor | None = None,
        relationships: dict[str, float] | None = None,
        basis_weights: torch.Tensor | None = None,
        composition: dict[str, float] | None = None,
        frequency: float = 1.0,
        phase: float = 0.0,
        energy: float = 0.0,
        **meta,
    ) -> Concept:
        """Add a concept to the graph with detailed logging."""
        if name in self.concepts:
            self._log_concept_operation(
                "Update existing concept",
                name,
                {
                    "energy": self.concepts[name].energy,
                    "new_embedding_shape": embedding.shape if embedding is not None else None
                }
            )
        else:
            self._log_concept_operation("Add new concept", name)
            
        # Create or update concept
        concept = Concept(
            name=name,
            embedding=embedding,
            energy=energy,
            basis_weights=basis_weights,
            composition_weights=composition or {},
            frequency=frequency,
            phase=phase,
            **meta
        )
        
        self.concepts[name] = concept
        self.adj[name] = []
        
        # Add to NetworkX graph
        self.graph.add_node(name, concept=concept)
        
        # Add relationships if provided
        if relationships:
            for target, weight in relationships.items():
                if target in self.concepts:
                    self.add_edge(name, target, weight)
                    
        return concept

    def compose_concept(
        self,
        name: str,
        components: dict[str, float],
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
        for comp, weight in zip(comp_concepts, components.values(), strict=False):
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

    def get_concept_embedding(self, name: str) -> torch.Tensor | None:
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
        self,
        name: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Get top-k related concepts by edge weight."""
        if name not in self.graph:
            return []

        edges = self.graph.edges(name, data=True)
        related = [(target, data["weight"]) for _, target, data in edges]
        return sorted(related, key=lambda x: x[1], reverse=True)[:top_k]

    def get_top_concepts(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get top-k concepts by PageRank centrality."""
        centrality = nx.pagerank(self.graph)
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def propagate_energy(self, steps: int = 2):
        """Propagate energy through the concept graph recursively for multiple hops (default 2)."""
        for _ in range(steps):
            new_energies = {name: concept.energy for name, concept in self.concepts.items()}
            for source, concept in self.concepts.items():
                if concept.energy != 0:
                    for conn in self.adj.get(source, []):
                        target = conn.target
                        if target in self.concepts:
                            # Energy transfer based on edge weight (support complex)
                            transfer = concept.energy * conn.weight * 0.5
                            new_energies[target] += transfer
                            new_energies[source] -= transfer * 0.1
            for name, energy in new_energies.items():
                # Clamp to [0, 1] for real part, but preserve complex type
                real = max(0.0, min(1.0, energy.real))
                imag = energy.imag
                self.concepts[name].energy = complex(real, imag)
            for name, concept in self.concepts.items():
                if name in self.graph:
                    self.graph.nodes[name]["concept"] = concept

    def decay(self):
        """Apply global decay to all concepts."""
        r = self.decay_rate
        for n in self.graph.nodes():
            self.graph.nodes[n]["concept"].energy *= r
        self.current_time += 1
        return self.current_time

    def update_n_cluster(self, min_energy: float, purge_threshold: int):
        """Update the number of clusters by pruning low-energy concepts."""
        # First, update access counts and energies
        for concept in self.concepts.values():
            concept.access_count += 1
        
        # Then prune based on energy and access thresholds
        self.prune(energy_thresh=min_energy, access_thresh=purge_threshold)

    def prune(self, energy_thresh: float, access_thresh: int = 0):
        """Prune concepts based on energy and access thresholds."""
        to_remove = []
        for name, concept in self.concepts.items():
            energy_val = concept.energy.real if isinstance(concept.energy, complex) else concept.energy
            if energy_val < energy_thresh and concept.access_count < access_thresh:
                to_remove.append(name)
        for name in to_remove:
            del self.concepts[name]
            if name in self.adj:
                del self.adj[name]
            self.graph.remove_node(name)

    def define_region(self, region_name: str, purpose: Any = None):
        """Define a new region in the concept graph."""
        if region_name not in self.regions:
            self.regions[region_name] = set()
            self.region_meta[region_name] = purpose

    def add_to_region(self, region_name: str, concept_name: str):
        """Add a concept to a region."""
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} does not exist")
        if concept_name not in self.concepts:
            raise ValueError(f"Concept {concept_name} does not exist")
            
        self.regions[region_name].add(concept_name)
        self.concepts[concept_name].regions.add(region_name)

    def get_region(self, region_name: str) -> set[str] | None:
        """Get all concepts in a region."""
        return self.regions.get(region_name)

    def to_bytecode(
        self,
        start_concept: str | None = None,
        max_hops: int = 5,
    ) -> list[tuple]:
        """Convert concept graph to bytecode with detailed logging.
        Ensures every concept referenced in a CBIND is asserted (CASSERT) before any CBIND referencing it.
        """
        if not self.concepts:
            self.logger.debug("No concepts to convert to bytecode")
            return []

        self._log_concept_operation(
            "Start bytecode generation",
            "all",
            {"start_concept": start_concept, "max_hops": max_hops},
        )

        visited = set()
        asserted = set()
        bytecode = []

        def assert_concept(concept_name):
            if concept_name not in asserted:
                bytecode.append((Opcode.CASSERT, [concept_name]))
                asserted.add(concept_name)

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

            # Assert this concept before any relationships
            assert_concept(node)

            # Process relationships - fix the unpacking issue
            for source, target, data in self.graph.edges(node, data=True):
                weight = data.get("weight", 1.0)
                if weight > 0.5:  # Only process significant relationships
                    # Assert the target concept before binding
                    assert_concept(target)
                    bytecode.append((Opcode.CBIND, [node, target, str(weight)]))
                    traverse(target, depth + 1)

        # Start traversal
        if start_concept:
            traverse(start_concept, 0)
        else:
            # Start with highest energy concepts
            sorted_concepts = sorted(
                self.concepts.items(),
                key=lambda x: x[1].energy.real if isinstance(x[1].energy, complex) else x[1].energy,
                reverse=True,
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

    def update_embeddings(self, embeddings: dict[str, torch.Tensor]):
        """Update concept embeddings."""
        for name, embedding in embeddings.items():
            if name in self.graph:
                self.graph.nodes[name]["concept"].embedding = embedding

    def get_subgraph(self, concepts: list[str]) -> nx.DiGraph:
        """Get subgraph containing only specified concepts."""
        return self.graph.subgraph(concepts).copy()

    def merge(self, other: ConceptGraph):
        """Merge another concept graph into this one."""
        # Merge concepts
        for name, concept in other.concepts.items():
            if name not in self.concepts:
                self.add_concept(
                    name=name,
                    embedding=concept.embedding,
                    energy=concept.energy,
                    basis_weights=concept.basis_weights,
                    composition=concept.composition_weights,
                    frequency=concept.frequency,
                    phase=concept.phase
                )
                
        # Merge edges
        for source, target, data in other.graph.edges(data=True):
            if source in self.concepts and target in self.concepts:
                self.add_edge(
                    source=source,
                    target=target,
                    weight=data.get('weight', 1.0),
                    relationship=data.get('relationship', 'co-occurrence')
                )

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

    def get_embeddings(self) -> dict[str, torch.Tensor]:
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

    def get_relationships(self) -> list[tuple[str, str, float]]:
        """Get all relationships in the graph."""
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            relationships.append((source, target, data.get('weight', 1.0)))
        return relationships

    def get_concept(self, name: str) -> Concept:
        """Return the Concept object for a given name."""
        if name in self.concepts:
            return self.concepts[name]
        raise KeyError(f"Concept '{name}' not found in ConceptGraph.")
