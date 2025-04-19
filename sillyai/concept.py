import asyncio
import logging
from collections import defaultdict
import heapq

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Connection:
    def __init__(self, target: str, weight: float = 1.0, relationship: str = "co-occurrence"):
        self.target = target
        self.weight = weight
        self.relationship = relationship

    def __repr__(self):
        return f"Connection(target={self.target!r}, weight={self.weight}, relationship={self.relationship!r})"

class Concept:
    def __init__(self, name: str, energy: float = 0.0):
        self.name = name
        self.energy = energy
        self.connections: list[Connection] = []
        self.access_count = 0
        self.last_access_time = 0

    def __repr__(self):
        conns = ', '.join([repr(c) for c in self.connections])
        return (
            f"Concept(name={self.name!r}, energy={self.energy:.3f}, "
            f"access_count={self.access_count}, last_access_time={self.last_access_time}, "
            f"connections=[{conns}])"
        )

class ConceptGraph:
    def __init__(self, decay_rate: float = 0.45, debug: bool = False):
        self.concepts: dict[str, Concept] = {}
        self.decay_rate = decay_rate
        self.time = 0
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def add_concept(self, name: str) -> None:
        if name not in self.concepts:
            self.concepts[name] = Concept(name)
            logger.debug(f"Added concept: {name}")

    def add_connection(self, source: str, target: str, weight: float = 1.0, relationship: str = "co-occurrence") -> None:
        self.add_concept(source)
        self.add_concept(target)
        self.concepts[source].connections.append(Connection(target, weight, relationship))
        self.concepts[target].connections.append(Connection(source, weight, relationship))
        logger.debug(f"Connected {source} <-> {target} with weight={weight}, relationship={relationship}")

    async def _propagate_energy(self, transfer_factor: float = 0.1) -> None:
        new_energy = defaultdict(float)
        # Gather tasks for parallel simulation
        for concept in list(self.concepts.values()):
            for conn in concept.connections:
                # calculate transfer
                transferred = concept.energy * conn.weight * transfer_factor
                new_energy[conn.target] += transferred
                new_energy[concept.name] -= transferred
        # Apply updates
        for name, delta in new_energy.items():
            if name in self.concepts:
                concept = self.concepts[name]
                old_energy = concept.energy
                concept.energy = max(0.0, old_energy + delta)
                self._update_access(name)
                logger.debug(f"Energy update for {name}: {old_energy:.3f} -> {concept.energy:.3f}")
                # allow other tasks to run
                await asyncio.sleep(0)

    def propagate_energy(self, transfer_factor: float = 0.1) -> None:
        """
        Synchronous wrapper for async energy propagation.
        """
        logger.info("Starting energy propagation...")
        try:
            asyncio.run(self._propagate_energy(transfer_factor))
        except RuntimeError:
            # If an event loop is already running (e.g., in notebooks), use create_task
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._propagate_energy(transfer_factor))
        logger.info("Energy propagation complete.")

    async def _update_n_cluster(self, min_energy: float = 0.01, purge_threshold: int = 5) -> None:
        # Decay and collect low-energy concepts
        to_remove = []
        for name, concept in list(self.concepts.items()):
            old_energy = concept.energy
            concept.energy *= self.decay_rate
            if concept.energy < min_energy:
                to_remove.append(name)
            logger.debug(f"Decayed {name}: {old_energy:.3f} -> {concept.energy:.3f}")
            await asyncio.sleep(0)
        # Remove low-energy
        for name in to_remove:
            self._purge_concept(name)
        # LFU-based purge
        heap = [(c.energy, c.access_count, c.name) for c in self.concepts.values()]
        heapq.heapify(heap)
        while len(self.concepts) > purge_threshold:
            energy, count, name = heapq.heappop(heap)
            if name in self.concepts:
                logger.debug(f"LFU purge concept: {name} with energy={energy:.3f}, access_count={count}")
                self._purge_concept(name)
            await asyncio.sleep(0)

    def update_n_cluster(self, min_energy: float = 0.01, purge_threshold: int = 5) -> None:
        """
        Synchronous wrapper for async cluster update (decay & purge).
        """
        logger.info("Starting cluster update (decay & LFU purge)...")
        try:
            asyncio.run(self._update_n_cluster(min_energy, purge_threshold))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._update_n_cluster(min_energy, purge_threshold))
        logger.info("Cluster update complete.")

    def _update_access(self, concept_name: str) -> None:
        concept = self.concepts[concept_name]
        concept.access_count += 1
        concept.last_access_time = self.time
        self.time += 1
        logger.debug(f"Access updated for {concept_name}: count={concept.access_count}, time={concept.last_access_time}")

    def _purge_concept(self, name: str) -> None:
        if name in self.concepts:
            del self.concepts[name]
            for concept in self.concepts.values():
                concept.connections = [c for c in concept.connections if c.target != name]
            logger.debug(f"Purged concept: {name}")

    def __str__(self) -> str:
        lines = []
        for name, c in self.concepts.items():
            conns = ', '.join(f"{conn.target}({conn.weight:.2f},{conn.relationship})" for conn in c.connections)
            lines.append(f"{name}: Energy={c.energy:.2f}, Access={c.access_count}, Connections=[{conns}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()
