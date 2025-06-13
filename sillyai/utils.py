import asyncio
import bisect
import hashlib
import logging
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

import numpy as np
import psutil
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ANSI color codes for Windows/Unix
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

CACHE_DIR = "cache"
CHUNK_SIZE = 1000  # windows per chunk
NPZ_EXT = ".npz"
MAX_WORKERS = min(
    psutil.cpu_count(logical=False),
    4,
)  # Limit workers for low-end devices
BATCH_SIZE = 32  # Smaller batch size for CPU training
PREFETCH_FACTOR = 2  # Number of batches to prefetch


class DatasetStats:
    """Track and display dataset statistics."""

    def __init__(self):
        self.total_samples = 0
        self.chunk_sizes = []
        self.load_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage = 0
        self.start_time = time.time()

    def update(self, chunk_size: int, load_time: float, cache_hit: bool):
        """Update statistics."""
        self.total_samples += chunk_size
        self.chunk_sizes.append(chunk_size)
        self.load_times.append(load_time)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print(f"{CYAN}📊 Dataset Statistics{RESET}")
        print("=" * 50)
        print(f"{GREEN}📈 Total Samples: {self.total_samples:,}")
        print(f"{BLUE}📦 Average Chunk Size: {np.mean(self.chunk_sizes):.1f}")
        print(f"{MAGENTA}⚡ Average Load Time: {np.mean(self.load_times) * 1000:.1f}ms")
        print(
            f"{YELLOW}💾 Cache Hit Rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%",
        )
        print(f"{RED}🧠 Memory Usage: {self.get_memory_usage():.1f}MB")
        print(f"{CYAN}⏱️  Total Time: {time.time() - self.start_time:.1f}s")
        print("=" * 50 + "\n")


class OptimizedChunkedMMapDataset(Dataset):
    """Optimized dataset with streaming, caching, and parallel loading."""

    def __init__(self, cache_dir: str, chunk_size: int = CHUNK_SIZE):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_offsets = [0]
        self.stats = DatasetStats()

        # Initialize thread pool for parallel loading
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Initialize LRU cache for frequently accessed chunks
        self.chunk_cache = {}
        self.cache_lock = Lock()

        # Initialize prefetch queue
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_FACTOR)
        self.prefetch_task = None

        # Initialize data hash map for deduplication
        self.data_hash_map = defaultdict(list)

        # Find all chunk files
        print(f"\n{CYAN}🔍 Scanning dataset chunks...{RESET}")
        for f in tqdm(os.listdir(cache_dir), desc="Loading chunks"):
            if f.endswith(".npy"):
                chunk_path = os.path.join(cache_dir, f)
                self.chunks.append(np.load(chunk_path, mmap_mode="r"))
                self.chunk_offsets.append(self.chunk_offsets[-1] + len(self.chunks[-1]))

        self.total_size = self.chunk_offsets[-1]
        print(
            f"\n{GREEN}✅ Loaded {len(self.chunks)} chunks with {self.total_size:,} total samples{RESET}",
        )

    def _hash_data(self, data: np.ndarray) -> str:
        """Generate hash for data deduplication."""
        return hashlib.md5(data.tobytes()).hexdigest()

    def _deduplicate_data(self, data: np.ndarray) -> np.ndarray:
        """Remove duplicate samples using hashing."""
        data_hash = self._hash_data(data)
        if data_hash in self.data_hash_map:
            return self.data_hash_map[data_hash][0]
        self.data_hash_map[data_hash].append(data)
        return data

    async def _prefetch_worker(self):
        """Background worker for prefetching chunks."""
        while True:
            try:
                # Predict next chunk to load
                next_chunk_idx = self._predict_next_chunk()

                # Load chunk asynchronously
                chunk_data = await self._load_chunk_async(next_chunk_idx)
                if chunk_data is not None:
                    await self.prefetch_queue.put((next_chunk_idx, chunk_data))

                await asyncio.sleep(0.1)  # Prevent tight loop
            except Exception as e:
                print(f"{RED}Error in prefetch worker: {e}{RESET}")
                await asyncio.sleep(1)

    def _predict_next_chunk(self) -> int:
        """Predict next chunk to load based on access patterns."""
        with self.cache_lock:
            if not self.chunk_cache:
                return 0
            # Return least recently used chunk
            return min(self.chunk_cache.items(), key=lambda x: x[1][1])[0]

    async def _load_chunk_async(self, chunk_idx: int) -> np.ndarray | None:
        """Load chunk asynchronously."""
        if chunk_idx >= len(self.chunks):
            return None

        try:
            # Run numpy load in thread pool
            loop = asyncio.get_event_loop()
            chunk_data = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.chunks[chunk_idx].copy(),
            )

            # Deduplicate data
            chunk_data = self._deduplicate_data(chunk_data)

            return chunk_data
        except Exception as e:
            print(f"{RED}Error loading chunk {chunk_idx}: {e}{RESET}")
            return None

    def __getitem__(
        self,
        idx: int | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """Get item(s) from dataset with optimized loading."""
        start_time = time.time()

        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]

        result = self._get_single_item(idx)

        # Update statistics
        self.stats.update(1, time.time() - start_time, idx in self.chunk_cache)

        return result

    def _get_single_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get single item with optimized loading."""
        if not 0 <= idx < self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")

        # Find which chunk contains this index
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]

        # Check cache first
        with self.cache_lock:
            if chunk_idx in self.chunk_cache:
                chunk_data = self.chunk_cache[chunk_idx][0]
                self.chunk_cache[chunk_idx] = (chunk_data, time.time())
                return torch.tensor(chunk_data[local_idx])

        # Load chunk if not in cache
        chunk_data = self.chunks[chunk_idx][local_idx]

        # Update cache
        with self.cache_lock:
            self.chunk_cache[chunk_idx] = (chunk_data, time.time())

        return torch.tensor(chunk_data)

    def __len__(self) -> int:
        return self.total_size

    def print_stats(self):
        """Print dataset statistics."""
        self.stats.print_stats()

    def close(self):
        """Clean up resources."""
        self.thread_pool.shutdown()
        if self.prefetch_task:
            self.prefetch_task.cancel()

    def __del__(self):
        self.close()


class OptimizedDataLoader(DataLoader):
    """Optimized DataLoader for CPU training."""

    def __init__(self, dataset, batch_size=BATCH_SIZE, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=MAX_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True,
            **kwargs,
        )

    def __iter__(self):
        """Create iterator with progress bar."""
        iterator = super().__iter__()
        return tqdm(iterator, desc="Training", total=len(self), unit="batch")

    def __del__(self):
        """Clean up resources."""
        try:
            self.dataset.close()
        except:
            pass


def get_or_create_loop():
    """Get event loop in a thread-safe way, creating if needed"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ensure_loop_running():
    """Ensure we have a running event loop, creating one if needed"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not loop.is_running() and threading.current_thread() is threading.main_thread():

        def run_loop():
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            except Exception:
                pass

        thread = threading.Thread(target=run_loop, daemon=True, name="AsyncIOThread")
        thread.start()

        # Give the loop time to start
        time.sleep(0.1)

    return loop


class AsyncInit:
    """Context manager to ensure proper async initialization"""

    def __init__(self):
        self.loop = None
        self.running_loop = False

    def __enter__(self):
        try:
            self.loop = asyncio.get_event_loop()
            self.running_loop = self.loop.is_running()
            if not self.running_loop:
                asyncio.set_event_loop(self.loop)
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.running_loop and self.loop and not self.loop.is_closed():
            self.loop.stop()
            self.loop.close()


class ChunkedMMapDataset(Dataset):
    """Dataset that loads data from memory-mapped files in chunks."""

    def __init__(self, cache_dir: str, chunk_size: int = CHUNK_SIZE):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_offsets = [0]
        self.stats = DatasetStats()

        # Initialize thread pool for parallel loading
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Initialize LRU cache for frequently accessed chunks
        self.chunk_cache = {}
        self.cache_lock = Lock()

        # Initialize prefetch queue
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_FACTOR)
        self.prefetch_task = None

        # Initialize data hash map for deduplication
        self.data_hash_map = defaultdict(list)

        # Find all chunk files
        print(f"\n{CYAN}🔍 Scanning dataset chunks...{RESET}")
        for f in tqdm(os.listdir(cache_dir), desc="Loading chunks"):
            if f.endswith(".npy"):
                chunk_path = os.path.join(cache_dir, f)
                self.chunks.append(np.load(chunk_path, mmap_mode="r"))
                self.chunk_offsets.append(self.chunk_offsets[-1] + len(self.chunks[-1]))

        self.total_size = self.chunk_offsets[-1]
        print(
            f"\n{GREEN}✅ Loaded {len(self.chunks)} chunks with {self.total_size:,} total samples{RESET}",
        )

    def _hash_data(self, data: np.ndarray) -> str:
        """Generate hash for data deduplication."""
        return hashlib.md5(data.tobytes()).hexdigest()

    def _deduplicate_data(self, data: np.ndarray) -> np.ndarray:
        """Remove duplicate samples using hashing."""
        data_hash = self._hash_data(data)
        if data_hash in self.data_hash_map:
            return self.data_hash_map[data_hash][0]
        self.data_hash_map[data_hash].append(data)
        return data

    async def _prefetch_worker(self):
        """Background worker for prefetching chunks."""
        while True:
            try:
                # Predict next chunk to load
                next_chunk_idx = self._predict_next_chunk()

                # Load chunk asynchronously
                chunk_data = await self._load_chunk_async(next_chunk_idx)
                if chunk_data is not None:
                    await self.prefetch_queue.put((next_chunk_idx, chunk_data))

                await asyncio.sleep(0.1)  # Prevent tight loop
            except Exception as e:
                print(f"{RED}Error in prefetch worker: {e}{RESET}")
                await asyncio.sleep(1)

    def _predict_next_chunk(self) -> int:
        """Predict next chunk to load based on access patterns."""
        with self.cache_lock:
            if not self.chunk_cache:
                return 0
            # Return least recently used chunk
            return min(self.chunk_cache.items(), key=lambda x: x[1][1])[0]

    async def _load_chunk_async(self, chunk_idx: int) -> np.ndarray | None:
        """Load chunk asynchronously."""
        if chunk_idx >= len(self.chunks):
            return None

        try:
            # Run numpy load in thread pool
            loop = asyncio.get_event_loop()
            chunk_data = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.chunks[chunk_idx].copy(),
            )

            # Deduplicate data
            chunk_data = self._deduplicate_data(chunk_data)

            return chunk_data
        except Exception as e:
            print(f"{RED}Error loading chunk {chunk_idx}: {e}{RESET}")
            return None

    def __getitem__(
        self,
        idx: int | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """Get item(s) from dataset with optimized loading."""
        start_time = time.time()

        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]

        result = self._get_single_item(idx)

        # Update statistics
        self.stats.update(1, time.time() - start_time, idx in self.chunk_cache)

        return result

    def _get_single_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get single item with optimized loading."""
        if not 0 <= idx < self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")

        # Find which chunk contains this index
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]

        # Check cache first
        with self.cache_lock:
            if chunk_idx in self.chunk_cache:
                chunk_data = self.chunk_cache[chunk_idx][0]
                self.chunk_cache[chunk_idx] = (chunk_data, time.time())
                return torch.tensor(chunk_data[local_idx])

        # Load chunk if not in cache
        chunk_data = self.chunks[chunk_idx][local_idx]

        # Update cache
        with self.cache_lock:
            self.chunk_cache[chunk_idx] = (chunk_data, time.time())

        return torch.tensor(chunk_data)

    def __len__(self) -> int:
        return self.total_size

    def print_stats(self):
        """Print dataset statistics."""
        self.stats.print_stats()

    def close(self):
        """Clean up resources."""
        self.thread_pool.shutdown()
        if self.prefetch_task:
            self.prefetch_task.cancel()

    def __del__(self):
        self.close()


class IterableMMapDataset(IterableDataset):
    """
    Fully streaming dataset with async prefetching. Compatible with both cached npz
    files and dynamically generated Schrödinger data.
    """

    def __init__(
        self,
        cache_dir=CACHE_DIR,
        prefetch_chunks=2,
        buffer_size=1000,
        seq_len=64,
        potential_type="harmonic",
        samples_per_chunk=CHUNK_SIZE,
    ):
        self.cache_dir = cache_dir
        self.prefetch_chunks = prefetch_chunks
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.potential_type = potential_type
        self.samples_per_chunk = samples_per_chunk

        # Try to use cached data first
        if os.path.exists(cache_dir):
            self.chunk_files = sorted(
                f for f in os.listdir(cache_dir) if f.endswith(NPZ_EXT)
            )
            if self.chunk_files:
                self.use_cache = True

                # Initialize basic components first
                self.sample_buffer = deque(maxlen=buffer_size)
                self.current_chunk = None
                self.chunk_idx = 0

                # Initialize async components with safe loop creation
                self.loop = get_or_create_loop()
                self.chunk_queue = asyncio.Queue(maxsize=prefetch_chunks)
                self.prefetch_task = None
                self.is_closing = False
                return

        # Fallback to dynamic generation
        self.use_cache = False
        self.x = np.linspace(-1, 1, seq_len).astype(np.float32)
        self.samples_generated = 0

    def _make_sample(self):
        """Generate a sample dynamically like SchrödingerDataset"""
        if self.potential_type == "harmonic":
            k = np.random.uniform(1.0, 5.0)
            V = 0.5 * k * self.x**2
            psi = np.exp(-np.sqrt(k) * self.x**2 / 2)
        else:
            V = np.zeros_like(self.x)
            psi = np.sin(np.pi * (self.x + 1) / 2)

        psi = psi / np.linalg.norm(psi)
        V = V.astype(np.float32)[:, None]
        psi = psi.astype(np.float32)[:, None]

        # Convert to complex tensors
        Vc = torch.complex(torch.from_numpy(V), torch.zeros_like(torch.from_numpy(V)))
        ψc = torch.complex(
            torch.from_numpy(psi),
            torch.zeros_like(torch.from_numpy(psi)),
        )
        return Vc, ψc

    def __iter__(self):
        if not self.use_cache:
            while True:
                yield self._make_sample()
                self.samples_generated += 1

        # Reset state and start prefetching for cached data
        self.is_closing = False
        self.chunk_idx = 0
        self.current_chunk = None
        self.sample_buffer.clear()

        # Start prefetch worker
        if self.prefetch_task is None or self.prefetch_task.done():
            self.prefetch_task = self.loop.create_task(self._prefetch_worker())

        return self

    def __next__(self):
        if not self.use_cache:
            return self._make_sample()

        if self.is_closing:
            raise StopIteration

        # Fill buffer if needed
        while not self.sample_buffer:
            if self.current_chunk is None:
                try:
                    # Non-blocking check for more chunks
                    if (
                        self.chunk_idx >= len(self.chunk_files)
                        and self.chunk_queue.empty()
                    ):
                        raise StopIteration

                    # Get next chunk with timeout
                    chunk_idx, chunk_data = self.loop.run_until_complete(
                        asyncio.wait_for(self.chunk_queue.get(), timeout=1.0),
                    )
                    self.current_chunk = chunk_data
                except TimeoutError:
                    if (
                        self.chunk_idx >= len(self.chunk_files)
                        and self.chunk_queue.empty()
                    ):
                        raise StopIteration
                    continue
                except Exception as e:
                    print(f"Error getting next chunk: {e}")
                    raise StopIteration

            # Process current chunk
            if self.current_chunk is not None:
                try:
                    # Convert to complex tensors
                    for i in range(len(self.current_chunk["x_real"])):
                        V = torch.complex(
                            torch.from_numpy(self.current_chunk["x_real"][i].copy()),
                            torch.from_numpy(self.current_chunk["x_imag"][i].copy()),
                        )
                        ψ = torch.complex(
                            torch.from_numpy(self.current_chunk["y_real"][i].copy()),
                            torch.from_numpy(self.current_chunk["y_imag"][i].copy()),
                        )
                        self.sample_buffer.append((V, ψ))

                    self.current_chunk = None
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    self.current_chunk = None
                    continue

        return self.sample_buffer.popleft()

    def close(self):
        if self.use_cache:
            self.is_closing = True
            if self.prefetch_task:
                self.prefetch_task.cancel()
                self.prefetch_task = None
            self.sample_buffer.clear()
            self.current_chunk = None

    def __del__(self):
        self.close()


class Logger:
    """Professional logging utility for SillyAI."""

    def __init__(self, logfile="debug.log"):
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Set up file logger
        self.logger = logging.getLogger("sillyai")
        self.logger.setLevel(logging.DEBUG)

        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Remove any 'logs/' prefix from logfile to avoid nested directories
        logfile = logfile.replace("logs/", "")
        log_file = os.path.join("logs", f"{logfile}_{timestamp}")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Initialize history tracking
        self.tensor_history = {}
        self.model_history = {}
        self.next_tensor_id = 0
        self.next_model_id = 0

    def clear_log(self):
        """Clear the log file."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def _write(self, msg, level="info"):
        """Write a message to the log."""
        # Map level to logging level
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        log_level = level_map.get(level.lower(), logging.INFO)

        # Log without extra fields to avoid conflicts
        self.logger.log(log_level, msg)

    def info(self, msg):
        """Log an info message."""
        self._write(msg, level="info")

    def warning(self, msg):
        """Log a warning message."""
        self._write(msg, level="warning")

    def error(self, msg):
        """Log an error message."""
        self._write(msg, level="error")

    def debug(self, msg):
        """Log a debug message."""
        self._write(msg, level="debug")

    def log_tensor_op(self, tensor: torch.Tensor, op_name: str, **kwargs):
        """Log a tensor operation."""
        stats = self._get_tensor_stats(tensor)
        msg = f"Tensor operation '{op_name}': {stats}"
        self._write(msg, level="debug")

    def log_model_state(self, model: torch.nn.Module, state_name: str, **kwargs):
        """Log model state."""
        msg = f"Model state '{state_name}': {model.__class__.__name__}"
        self._write(msg, level="debug")

    def log_forward_pass(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """Log a forward pass."""
        msg = f"Forward pass in {module.__class__.__name__}"
        self._write(msg, level="debug")

    def log_backward_pass(
        self,
        module: torch.nn.Module,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ):
        """Log a backward pass."""
        msg = f"Backward pass in {module.__class__.__name__}"
        self._write(msg, level="debug")

    def log_optimizer_step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        """Log an optimizer step."""
        msg = f"Optimizer step: loss = {loss.item():.4f}"
        self._write(msg, level="debug")

    def log_concept_graph(self, graph, operation: str):
        """Log concept graph operation."""
        msg = f"Concept graph {operation}"
        self._write(msg, level="debug")


# Create global logger instance
logger = Logger()


def log_tensor(tensor: torch.Tensor, op_name: str, **kwargs):
    """Convenience function for logging tensor operations."""
    logger.log_tensor_op(tensor, op_name, **kwargs)


def log_model(model: torch.nn.Module, state_name: str, **kwargs):
    """Convenience function for logging model states."""
    logger.log_model_state(model, state_name, **kwargs)


def log_tensor(tensor, op_name, component="TensorOp", method=""):
    """Log tensor operation, shape, dtype, and device."""
    if isinstance(tensor, torch.Tensor):
        logger._write(
            component,
            method or op_name,
            f"{op_name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}",
        )
    elif isinstance(tensor, (tuple, list)):
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                logger._write(
                    component,
                    method or op_name,
                    f"{op_name}[{i}]: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}",
                )
