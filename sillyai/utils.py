import os
import time
import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from collections import deque
import sys
from datetime import datetime
from threading import Lock
from typing import Union, List, Tuple
import bisect

from datasets import Dataset

import torch
from torch.utils.data import IterableDataset
import numpy as np

# ANSI color codes for Windows/Unix
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
CACHE_DIR = "cache_chunks"
CHUNK_SIZE = 1_000   # windows per chunk
NPZ_EXT    = ".npz"


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
    
    def __init__(self, cache_dir: str, chunk_size: int = 1000):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_offsets = [0]
        
        # Find all chunk files
        for f in os.listdir(cache_dir):
            if f.endswith('.npy'):
                chunk_path = os.path.join(cache_dir, f)
                self.chunks.append(np.load(chunk_path, mmap_mode='r'))
                self.chunk_offsets.append(self.chunk_offsets[-1] + len(self.chunks[-1]))
                
        self.total_size = self.chunk_offsets[-1]
        
    def __len__(self) -> int:
        return self.total_size
        
    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        if isinstance(idx, list):
            # Handle batch indices
            return [self._get_single_item(i) for i in idx]
        return self._get_single_item(idx)
        
    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        if not 0 <= idx < self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
            
        # Find which chunk contains this index
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]
        
        # Get data from chunk
        chunk = self.chunks[chunk_idx]
        V, psi = chunk[local_idx]
        
        return torch.tensor(V), torch.tensor(psi)

    def _build_index(self):
        """Build the chunk index synchronously"""
        self.chunk_lens = []
        for fn in self.chunk_files:
            length = self._read_chunk_meta(fn)
            if length is not None:
                self.chunk_lens.append(length)
        
        self.cum_lens = np.cumsum([0] + self.chunk_lens)
        self.total = int(self.cum_lens[-1])
        
    def _read_chunk_meta(self, filename):
        """Read chunk metadata without loading full data"""
        try:
            with np.load(os.path.join(self.cache_dir, filename)) as f:
                return len(f['x_real'])  # Use x_real since we know this exists
        except Exception as e:
            print(f"Error reading chunk metadata from {filename}: {e}")
            return None

    async def _load_chunk_async(self, chunk_idx):
        """Load a chunk asynchronously"""
        if chunk_idx >= len(self.chunk_files):
            return None
            
        filename = self.chunk_files[chunk_idx]
        filepath = os.path.join(self.cache_dir, filename)
        
        try:            # Run numpy load in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: np.load(filepath))
            
            # Convert split complex data back to tensors
            V = torch.complex(
                torch.from_numpy(data['x_real']),
                torch.from_numpy(data['x_imag'])
            )
            psi = torch.complex(
                torch.from_numpy(data['y_real']),
                torch.from_numpy(data['y_imag'])
            )
            return [(V[i], psi[i]) for i in range(len(V))]
        except Exception as e:
            print(f"Error loading chunk {chunk_idx} from {filepath}: {e}")
            return None

    def _start_prefetching(self):
        """Start the prefetching task"""
        if self.prefetch_task is None:
            self.prefetch_task = asyncio.run_coroutine_threadsafe(
                self._prefetch_loop(), 
                self.loop
            )

    async def _prefetch_loop(self):
        """Main prefetch loop"""
        while not self.is_closing:
            try:
                # Predict next chunks to prefetch
                next_chunks = self._predict_next_chunks()
                
                for chunk_idx in next_chunks:
                    if self.is_closing:
                        break
                        
                    with self.cache_lock:
                        if chunk_idx not in self.chunk_cache:
                            data = await self._load_chunk_async(chunk_idx)
                            if data is not None:
                                self._update_cache(chunk_idx, data)
                
                await asyncio.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                print(f"Error in prefetch loop: {e}")
                await asyncio.sleep(1)  # Back off on error

    def _predict_next_chunks(self):
        """Predict which chunks to prefetch next based on access patterns"""
        with self.cache_lock:
            recent = sorted(
                self.access_times.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.prefetch_size]
            
        recent_chunks = [chunk_idx for chunk_idx, _ in recent]
        next_chunks = [
            (chunk_idx + 1) % len(self.chunk_files)
            for chunk_idx in recent_chunks
        ]
        return next_chunks

    def _update_cache(self, chunk_idx, data):
        """Update the chunk cache with new data"""
        while len(self.chunk_cache) >= self.max_cache_size:
            oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.chunk_cache[oldest]
            del self.access_times[oldest]
            
        self.chunk_cache[chunk_idx] = data
        self.access_times[chunk_idx] = time.time()

    def _get_chunk_sync(self, chunk_idx):
        """Get a chunk synchronously, falling back to sync load if needed"""
        with self.cache_lock:
            if chunk_idx in self.chunk_cache:
                self.access_times[chunk_idx] = time.time()
                return self.chunk_cache[chunk_idx]

        # Fall back to synchronous load if not in cache
        try:
            filename = self.chunk_files[chunk_idx]
            filepath = os.path.join(self.cache_dir, filename)
            data = np.load(filepath)
            
            # Convert split complex data back to tensors
            V = torch.complex(
                torch.from_numpy(data['x_real']),
                torch.from_numpy(data['x_imag'])
            )
            psi = torch.complex(
                torch.from_numpy(data['y_real']),
                torch.from_numpy(data['y_imag'])
            )
            samples = [(V[i], psi[i]) for i in range(len(V))]
            
            with self.cache_lock:
                self._update_cache(chunk_idx, samples)
                
            return samples
        except Exception as e:
            print(f"Error loading chunk {chunk_idx} synchronously: {e}")
            return None

    def _make_sample(self):
        """Generate a sample dynamically like SchrödingerDataset"""
        if self.potential_type == 'harmonic':
            k = np.random.uniform(1.0, 5.0)
            V = 0.5 * k * self.x**2
            psi = np.exp(-np.sqrt(k) * self.x**2/2)
        else:
            V = np.zeros_like(self.x)
            psi = np.sin(np.pi * (self.x + 1)/2)
            
        psi = psi / np.linalg.norm(psi)
        V = V.astype(np.float32)[:, None]
        psi = psi.astype(np.float32)[:, None]
        
        # Convert to complex tensors like SchrödingerDataset
        Vc = torch.complex(torch.from_numpy(V), torch.zeros_like(torch.from_numpy(V)))
        ψc = torch.complex(torch.from_numpy(psi), torch.zeros_like(torch.from_numpy(psi)))
        return Vc, ψc

    def __getitem__(self, idx):
        # Only accept a single integer index
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"ChunkedMMapDataset only supports integer indices, got {type(idx)}: {idx}")
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        """Helper method to get a single item by integer index"""
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Index must be integer, got {type(idx)}")

        if not self.use_cache:
            if idx < 0 or idx >= self.total:
                raise IndexError(f"Index {idx} out of bounds")
            return self._make_sample()
        
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of bounds")

        chunk_idx = np.searchsorted(self.cum_lens[1:], idx, side='right')
        offset = idx - self.cum_lens[chunk_idx]
        
        chunk_data = self._get_chunk_sync(chunk_idx)
        if chunk_data is None:
            raise RuntimeError(f"Failed to load chunk {chunk_idx}")

        return chunk_data[offset]

    def __len__(self):
        if not self.use_cache:
            return self.total
        return self.total
        
    @staticmethod
    def prepare_cache(dataset_size=10000, chunk_size=CHUNK_SIZE, cache_dir=CACHE_DIR,
                     seq_len=64, potential_type='harmonic'):
        """
        Pre-generate and cache samples to disk in .npz format.
        
        Args:
            dataset_size (int): Total number of samples to generate
            chunk_size (int): Number of samples per chunk
            cache_dir (str): Directory to save chunks
            seq_len (int): Sequence length for samples
            potential_type (str): Type of potential to generate
        """
        os.makedirs(cache_dir, exist_ok=True)
        x = np.linspace(-1, 1, seq_len).astype(np.float32)
        
        chunks = dataset_size // chunk_size
        for chunk_idx in range(chunks):
            V_batch = []
            psi_batch = []
            
            for _ in range(chunk_size):
                if potential_type == 'harmonic':
                    k = np.random.uniform(1.0, 5.0)
                    V = 0.5 * k * x**2
                    psi = np.exp(-np.sqrt(k) * x**2/2)
                else:
                    V = np.zeros_like(x)
                    psi = np.sin(np.pi * (x + 1)/2)
                    
                psi = psi / np.linalg.norm(psi)
                V = V.astype(np.float32)[:, None]
                psi = psi.astype(np.float32)[:, None]
                
                V_batch.append(V)
                psi_batch.append(psi)
            
            # Save as complex arrays split into real/imag parts
            V_arr = np.stack(V_batch)
            psi_arr = np.stack(psi_batch)
            
            chunk_path = os.path.join(cache_dir, f'chunk_{chunk_idx:04d}.npz')
            np.savez(chunk_path,
                     x_real=V_arr.real,
                     x_imag=V_arr.imag,
                     y_real=psi_arr.real,
                     y_imag=psi_arr.imag)
            
            print(f'\rSaved chunk {chunk_idx + 1}/{chunks}', end='', flush=True)
        print('\nDone caching dataset')


class IterableMMapDataset(IterableDataset):
    """
    Fully streaming dataset with async prefetching. Compatible with both cached npz 
    files and dynamically generated Schrödinger data.
    """
    def __init__(self, cache_dir=CACHE_DIR, prefetch_chunks=2, buffer_size=1000,
                 seq_len=64, potential_type='harmonic', samples_per_chunk=CHUNK_SIZE):
        self.cache_dir = cache_dir
        self.prefetch_chunks = prefetch_chunks
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.potential_type = potential_type
        self.samples_per_chunk = samples_per_chunk
        
        # Try to use cached data first
        if os.path.exists(cache_dir):
            self.chunk_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(NPZ_EXT))
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
        if self.potential_type == 'harmonic':
            k = np.random.uniform(1.0, 5.0)
            V = 0.5 * k * self.x**2
            psi = np.exp(-np.sqrt(k) * self.x**2/2)
        else:
            V = np.zeros_like(self.x)
            psi = np.sin(np.pi * (self.x + 1)/2)
            
        psi = psi / np.linalg.norm(psi)
        V = V.astype(np.float32)[:, None]
        psi = psi.astype(np.float32)[:, None]
        
        # Convert to complex tensors
        Vc = torch.complex(torch.from_numpy(V), torch.zeros_like(torch.from_numpy(V)))
        ψc = torch.complex(torch.from_numpy(psi), torch.zeros_like(torch.from_numpy(psi)))
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
                    if self.chunk_idx >= len(self.chunk_files) and self.chunk_queue.empty():
                        raise StopIteration
                    
                    # Get next chunk with timeout
                    chunk_idx, chunk_data = self.loop.run_until_complete(
                        asyncio.wait_for(self.chunk_queue.get(), timeout=1.0)
                    )
                    self.current_chunk = chunk_data
                except asyncio.TimeoutError:
                    if self.chunk_idx >= len(self.chunk_files) and self.chunk_queue.empty():
                        raise StopIteration
                    continue
                except Exception as e:
                    print(f"Error getting next chunk: {e}")
                    raise StopIteration
            
            # Process current chunk
            if self.current_chunk is not None:
                try:
                    # Convert to complex tensors
                    for i in range(len(self.current_chunk['x_real'])):
                        V = torch.complex(
                            torch.from_numpy(self.current_chunk['x_real'][i].copy()),
                            torch.from_numpy(self.current_chunk['x_imag'][i].copy())
                        )
                        ψ = torch.complex(
                            torch.from_numpy(self.current_chunk['y_real'][i].copy()),
                            torch.from_numpy(self.current_chunk['y_imag'][i].copy())
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
    def __init__(self, logfile='debug.log'):
        self.logfile = logfile
        self.lock = Lock()

    def clear_log(self):
        """Overwrite the log file (clear contents)."""
        with self.lock:
            with open(self.logfile, 'w', encoding='utf-8') as f:
                f.write('')

    def _write(self, msg):
        with self.lock:
            with open(self.logfile, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    def log(self, component, method, msg, level='info'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prefix = f'({timestamp}) [{component}::{method}]'
        color = {'info': BLUE, 'warning': YELLOW, 'error': RED}.get(level, '')
        reset = RESET if color else ''
        formatted = f'{prefix} {msg}'
        # Print colored to console
        print(f'{color}{formatted}{reset}', file=sys.stderr if level=='error' else sys.stdout)
        # Write plain to log file
        self._write(formatted)

    def info(self, component, method, msg):
        self.log(component, method, msg, 'info')
    def warning(self, component, method, msg):
        self.log(component, method, msg, 'warning')
    def error(self, component, method, msg):
        self.log(component, method, msg, 'error')

def log_tensor(tensor, op_name, component="TensorOp", method=""): 
    """Log tensor operation, shape, dtype, and device."""
    if isinstance(tensor, torch.Tensor):
        logger.info(component, method or op_name, f"{op_name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    elif isinstance(tensor, (tuple, list)):
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                logger.info(component, method or op_name, f"{op_name}[{i}]: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

logger = Logger()