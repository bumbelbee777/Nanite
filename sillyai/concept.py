from collections import defaultdict
import heapq

class Connection:
    def __init__(self, target, weight, relationship):
        self.target = target
        self.weight = weight
        self.relationship = relationship

class Concept:
    def __init__(self, name, energy=0.0):
        self.name = name
        self.energy = energy
        self.connections = []
        self.access_count = 0  # Count of accesses for LFU
        self.last_access_time = 0  # Tracks the last time this concept was accessed

class ConceptGraph:
    def __init__(self, decay_rate=0.45):
        self.concepts = defaultdict(Concept)
        self.decay_rate = decay_rate
        self.time = 0  # To keep track of the access time for LFU

    def add_concept(self, name):
        self.concepts[name] = Concept(name)

    def add_connection(self, source, target, weight=1.0, relationship="co-occurrence"):
        if source not in self.concepts:
            self.add_concept(source)
        if target not in self.concepts:
            self.add_concept(target)
        
        self.concepts[source].connections.append(
            Connection(target, weight, relationship))
        self.concepts[target].connections.append(
            Connection(source, weight, relationship))

    def propagate_energy(self, transfer_factor=0.1):
        new_energy = defaultdict(float)
        for concept in self.concepts.values():
            for conn in concept.connections:
                transferred = concept.energy * conn.weight * transfer_factor
                new_energy[conn.target] += transferred
                new_energy[concept.name] -= transferred
        
        # Update energy and access count
        for name, delta in new_energy.items():
            if name in self.concepts:
                self.concepts[name].energy = max(0, self.concepts[name].energy + delta)
                self._update_access(name)

    def _update_access(self, concept_name):
        """ Update access count and time for LFU. """
        concept = self.concepts[concept_name]
        concept.access_count += 1
        concept.last_access_time = self.time
        self.time += 1

    def update_n_cluster(self, min_energy=0.01, purge_threshold=5):
        """ Apply decay, remove low-energy concepts, and implement LFU purge. """
        to_remove = []

        # Apply decay and determine low-energy concepts
        for name, concept in list(self.concepts.items()):
            concept.energy *= self.decay_rate
            if concept.energy < min_energy:
                to_remove.append(name)

        # Remove low-energy concepts and cleanup connections
        for name in to_remove:
            del self.concepts[name]
            for concept in self.concepts.values():
                concept.connections = [
                    conn for conn in concept.connections
                    if conn.target != name
                ]

        # Now handle LFU purging - find concepts with low access count and low energy
        low_energy_and_access = []

        for concept in self.concepts.values():
            heapq.heappush(low_energy_and_access, (concept.energy, concept.access_count, concept.name))

        # Remove concepts that have both low energy and access count
        while len(low_energy_and_access) > purge_threshold:
            _, _, concept_name = heapq.heappop(low_energy_and_access)
            if concept_name in self.concepts:
                del self.concepts[concept_name]
                for concept in self.concepts.values():
                    concept.connections = [
                        conn for conn in concept.connections
                        if conn.target != concept_name
                    ]
    
    def __str__(self):
        return "\n".join(
            f"{name} (Energy: {c.energy:.2f}, Access Count: {c.access_count}): "
            f"{[f'{conn.target}({conn.weight}, {conn.relationship})' for conn in c.connections]}"
            for name, c in self.concepts.items()
        )

    def __repr__(self):
        return self.__str__()