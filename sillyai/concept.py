from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Optional, Dict, Set, List, Tuple, Any
import random, logging, copy
import torch
import networkx as nx
import numpy as np

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
    """Manages a graph of concepts and their relationships."""
    
    def __init__(self, max_size: int = 1000, decay_rate: float = 0.1):
        self.graph = nx.DiGraph()
        self.max_size = max_size
        self.concept_embeddings = {}
        self.decay_rate = decay_rate
        self.time = 0
        self.regions = {}
        self.region_meta = {}
        
    def add_concept(self, name: str, embedding: Optional[torch.Tensor] = None, 
                   relationships: Optional[Dict[str, float]] = None, **meta):
        """Add a concept to the graph with optional embedding and relationships."""
        if len(self.graph) >= self.max_size:
            # Remove least central concept if at capacity
            centrality = nx.pagerank(self.graph)
            least_central = min(centrality.items(), key=lambda x: x[1])[0]
            self.graph.remove_node(least_central)
            del self.concept_embeddings[least_central]
            
        # Add concept node
        self.graph.add_node(name, **meta)
        if embedding is not None:
            self.concept_embeddings[name] = embedding
            
        # Add relationships
        if relationships:
            for target, weight in relationships.items():
                if target in self.graph:
                    self.graph.add_edge(name, target, weight=weight)
                    
    def get_concept_embedding(self, name: str) -> Optional[torch.Tensor]:
        """Get the embedding for a concept."""
        return self.concept_embeddings.get(name)
        
    def get_related_concepts(self, name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k related concepts by edge weight."""
        if name not in self.graph:
            return []
            
        edges = self.graph.edges(name, data=True)
        related = [(target, data['weight']) for _, target, data in edges]
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
            e = self.graph.nodes[n].get('energy', 0.0)
            if e <= 0.0:
                continue
            retained = e * decay
            new_e[n] += retained
            prop_total = e - retained
            edges = self.graph.edges(n, data=True)
            if not edges:
                new_e[n] += prop_total
            else:
                wsum = sum(data['weight'] for _, _, data in edges)
                if wsum > 0:
                    factor = prop_total / wsum
                    for _, target, data in edges:
                        new_e[target] += data['weight'] * factor
        for n, e in new_e.items():
            self.graph.nodes[n]['energy'] = e
            
    def decay(self):
        """Apply global decay to all concepts."""
        r = self.decay_rate
        for n in self.graph.nodes():
            self.graph.nodes[n]['energy'] *= r
        self.time += 1
        return self.time
        
    def prune(self, energy_thresh: float, access_thresh: int = 0):
        """Prune concepts below energy and access thresholds."""
        to_remove = []
        for n in self.graph.nodes():
            node_data = self.graph.nodes[n]
            if (node_data.get('energy', 0.0) < energy_thresh and 
                node_data.get('access_count', 0) < access_thresh):
                to_remove.append(n)
        for n in to_remove:
            self.graph.remove_node(n)
            if n in self.concept_embeddings:
                del self.concept_embeddings[n]
                
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
        if 'regions' not in self.graph.nodes[concept_name]:
            self.graph.nodes[concept_name]['regions'] = set()
        self.graph.nodes[concept_name]['regions'].add(region_name)
        
    def get_region(self, region_name: str) -> Optional[Set[str]]:
        """Get all concepts in a region."""
        return self.regions.get(region_name)
        
    def to_bytecode(self, start_concept: Optional[str] = None, max_hops: int = 5) -> List[tuple]:
        """Convert concept graph to bytecode instructions."""
        if not self.graph:
            return []
            
        # If no start concept specified, use most central
        if start_concept is None:
            centrality = nx.pagerank(self.graph)
            start_concept = max(centrality.items(), key=lambda x: x[1])[0]
            
        bytecode = []
        visited = set()
        
        def traverse(node: str, depth: int):
            if depth > max_hops or node in visited:
                return
                
            visited.add(node)
            bytecode.append(('CONCEPT', node))
            
            # Add relationships as operations
            for target, data in self.graph.edges(node, data=True):
                weight = data['weight']
                bytecode.append(('RELATE', target, weight))
                traverse(target, depth + 1)
                
        traverse(start_concept, 0)
        return bytecode
        
    def update_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """Update concept embeddings."""
        for name, embedding in embeddings.items():
            if name in self.graph:
                self.concept_embeddings[name] = embedding
                
    def get_subgraph(self, concepts: List[str]) -> nx.DiGraph:
        """Get subgraph containing only specified concepts."""
        return self.graph.subgraph(concepts).copy()
        
    def merge(self, other: 'ConceptGraph'):
        """Merge another concept graph into this one."""
        for node in other.graph.nodes():
            if len(self.graph) < self.max_size:
                self.add_concept(node, **other.graph.nodes[node])
                
        for u, v, data in other.graph.edges(data=True):
            if u in self.graph and v in self.graph:
                self.graph.add_edge(u, v, weight=data['weight'])
                
        # Merge embeddings
        for name, embedding in other.concept_embeddings.items():
            if name in self.graph:
                self.concept_embeddings[name] = embedding
                
    def save(self, path: str):
        """Save concept graph to file."""
        state = {
            'graph': self.graph,
            'embeddings': self.concept_embeddings,
            'max_size': self.max_size,
            'decay_rate': self.decay_rate,
            'time': self.time,
            'regions': self.regions,
            'region_meta': self.region_meta
        }
        torch.save(state, path)
        
    def load(self, path: str):
        """Load concept graph from file."""
        state = torch.load(path)
        self.graph = state['graph']
        self.concept_embeddings = state['embeddings']
        self.max_size = state['max_size']
        self.decay_rate = state['decay_rate']
        self.time = state['time']
        self.regions = state['regions']
        self.region_meta = state['region_meta']