import pytest
from sillyai.graph.concept import ConceptGraph, Concept

def test_add_concept():
    graph = ConceptGraph()
    graph.add_concept("apple")
    assert "apple" in graph.concepts
    assert graph.concepts["apple"].energy == 0
    assert graph.concepts["apple"].last_access_time == 0
    assert graph.concepts["apple"].access_count == 0

def test_add_connection():
    graph = ConceptGraph()
    graph.add_concept("apple")
    graph.add_concept("fruit")
    graph.add_connection("apple", "fruit")
    assert any(conn.target == "fruit" for conn in graph.concepts["apple"].connections)
    assert any(conn.target == "apple" for conn in graph.concepts["fruit"].connections)

def test_propagate_energy():
    graph = ConceptGraph()
    graph.add_concept("apple")
    graph.add_concept("fruit")
    graph.add_concept("red")
    graph.add_connection("apple", "fruit")
    graph.add_connection("fruit", "red")
    graph.concepts["apple"].energy = 1.0
    graph.propagate_energy()
    assert graph.concepts["fruit"].energy > 0
    assert graph.concepts["red"].energy > 0
    assert graph.concepts["apple"].energy < 1.0

def test_update_access():
    graph = ConceptGraph()
    graph.add_concept("apple")
    graph._update_access("apple")
    assert graph.concepts["apple"].access_count == 1
    assert graph.concepts["apple"].last_access_time == 0
    graph._update_access("apple")
    assert graph.concepts["apple"].access_count == 2
    assert graph.concepts["apple"].last_access_time == 1

def test_update_n_cluster():
    graph = ConceptGraph()
    graph.add_concept("A")
    graph.add_concept("B")
    graph.add_concept("C")
    graph.add_connection("A", "B")
    graph.add_connection("B", "C")
    graph.concepts["A"].energy = 1.0
    graph.propagate_energy()
    graph._update_access("A")  # Increase access for 'A'
    graph.update_n_cluster(min_energy=0.1, purge_threshold=2)  # Assuming at least 3 concepts, keeping top 2
    assert len(graph.concepts) <= 2  # Should have pruned at least one

def test_example_concepts():
    graph = ConceptGraph(decay_rate=0.8)
    graph.add_concept("apple")
    graph.add_concept("fruit")
    graph.add_concept("red")
    
    graph._update_access("apple")
    graph._update_access("fruit")
    graph._update_access("red")  # Red is most frequently accessed
    
    graph.add_connection("apple", "fruit", weight=0.8, relationship="is_a")
    graph.add_connection("fruit", "red", weight=0.5, relationship="color")
    
    graph.concepts["apple"].energy = 1.0
    graph.propagate_energy()
    
    assert graph.concepts["fruit"].energy > 0
    assert graph.concepts["red"].energy > 0
    
    graph.update_n_cluster(min_energy=0.1, purge_threshold=2)
    
    assert "red" in graph.concepts
    assert len(graph.concepts) >= 1
    
    graph.add_concept("sweet")
    graph.add_connection("apple", "sweet", weight=0.7, relationship="taste")
    assert any(conn.target == "sweet" for conn in graph.concepts["apple"].connections)
    
    graph.propagate_energy()
    graph.update_n_cluster(min_energy=0.05, purge_threshold=2)
    assert len(graph.concepts) <= 3