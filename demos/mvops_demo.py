from datetime import datetime
from typing import List
import time
import torch
import random
import os
import matplotlib.pyplot as plt
import psutil
from matplotlib.animation import PillowWriter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import multiprocessing as mp
from numba import jit
import concurrent.futures

from nanite.ops import MultivectorOps

PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

CACHE_DIR = "cache_chunks"
os.makedirs(CACHE_DIR, exist_ok=True)

# Highly optimized batch tensor generation with ThreadPoolExecutor and NumPy

def _gen_chunk_np(args):
    chunk_count, shape, dtype = args
    real = np.random.randn(chunk_count, *shape).astype(np.float32)
    imag = np.random.randn(chunk_count, *shape).astype(np.float32)
    return real, imag

def generate_tensors(count: int, shape=(32, 32), dtype=torch.float32, chunk_size: int = 10000) -> torch.Tensor:
    """Generate a batch of random complex tensors using ThreadPoolExecutor and NumPy."""
    num_chunks = (count + chunk_size - 1) // chunk_size
    args = [(min(chunk_size, count - i * chunk_size), shape, dtype)
            for i in range(num_chunks)]
    with ThreadPoolExecutor(max_workers=min(num_chunks, os.cpu_count() or 4)) as pool:
        results = list(pool.map(_gen_chunk_np, args))
    real = np.concatenate([r[0] for r in results], axis=0)[:count]
    imag = np.concatenate([r[1] for r in results], axis=0)[:count]
    real_t = torch.from_numpy(real).to(dtype)
    imag_t = torch.from_numpy(imag).to(dtype)
    return torch.complex(real_t, imag_t)

# Helper: Save tensors to cache

def save_tensors_to_cache(tensors, cache_dir, txt_path, batch_size=1000):
    count = tensors.shape[0]
    npz_paths = []
    with open(txt_path, "w") as txtf:
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch = tensors[batch_start:batch_end].cpu().numpy()
            npz_path = os.path.join(cache_dir, f"tensor_batch_{batch_start:07d}_{batch_end:07d}.npz")
            np.savez_compressed(npz_path, batch=batch)
            npz_paths.append(npz_path)
            # Save as plaintext (real, imag)
            for i in range(batch.shape[0]):
                arr = batch[i]
                txtf.write(f"Tensor {batch_start + i}\n")
                if np.iscomplexobj(arr):
                    for row in arr:
                        txtf.write(" ".join(f"({v.real:.6g},{v.imag:.6g})" for v in row) + "\n")
                else:
                    for row in arr:
                        txtf.write(" ".join(f"{v:.6g}" for v in row) + "\n")
                txtf.write("\n")
    return npz_paths

def _load_npz_tensor(npz_path, idx, shape):
    with np.load(npz_path, mmap_mode='r') as data:
        batch = data['batch']
        return [torch.from_numpy(batch[i]) for i in range(batch.shape[0])]

def load_tensors_from_cache(count, shape, cache_dir, batch_size=1000):
    npz_files = [os.path.join(cache_dir, f"tensor_batch_{i:07d}_{min(i+batch_size, count):07d}.npz")
                 for i in range(0, count, batch_size)]
    tensors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(npz_files))) as executor:
        results = list(executor.map(lambda npz: _load_npz_tensor(npz, 0, shape), npz_files))
    for batch in results:
        tensors.extend(batch)
    return torch.stack(tensors)

if __name__ == "__main__":
    NUM_TENSORS = 1_000_000
    TENSOR_SHAPE = (16, 16)
    TXT_PATH = os.path.join(CACHE_DIR, f"tensors_{NUM_TENSORS}_{TENSOR_SHAPE[0]}x{TENSOR_SHAPE[1]}.txt")
    batch_size = 1000
    npz_files_exist = all(os.path.exists(os.path.join(CACHE_DIR, f"tensor_batch_{i:07d}_{min(i+batch_size, NUM_TENSORS):07d}.npz")) for i in range(0, NUM_TENSORS, batch_size))
    if npz_files_exist:
        print(f"Loading {NUM_TENSORS} tensors from cache (parallel, memmap, compressed)...")
        tensors = load_tensors_from_cache(NUM_TENSORS, TENSOR_SHAPE, CACHE_DIR, batch_size=batch_size)
    else:
        print(f"Generating {NUM_TENSORS} random complex tensors of shape {TENSOR_SHAPE} (complex64)...")
        tensors = generate_tensors(NUM_TENSORS, TENSOR_SHAPE)
        print("Saving tensors to cache (compressed batches)...")
        save_tensors_to_cache(tensors, CACHE_DIR, TXT_PATH, batch_size=batch_size)
    print("Done loading/generating tensors.")

    mvops = MultivectorOps()

    # Prepare pairs for benchmarking
    a_batch = tensors[0:NUM_TENSORS-1:2]
    b_batch = tensors[1:NUM_TENSORS:2]
    num_ops = a_batch.shape[0]

    stats = {"matmul": [], "mvops": []}
    cpu_stats = {"matmul": [], "mvops": []}
    ram_stats = {"matmul": [], "mvops": []}
    ops_stats = []
    gif_frames = []
    process = psutil.Process(os.getpid())

    def benchmark_op(name, op_func, a_batch, b_batch):
        print(f"Benchmarking {name}...")
        op_times = []
        cpu_usages = []
        ram_usages = []
        start = time.time()
        for i in range(num_ops):
            t0 = time.time()
            _ = op_func(a_batch[i], b_batch[i])
            t1 = time.time()
            op_times.append((t1 - t0) * 1000)  # ms
            if (i+1) % 1000 == 0:
                cpu = psutil.cpu_percent(interval=0.01)
                ram = process.memory_info().rss / (1024 ** 3)
                cpu_usages.append(cpu)
                ram_usages.append(ram)
                print(f"{name}: {i+1}/{num_ops} ops | Last op: {op_times[-1]:.3f} ms | CPU: {cpu:.1f}% | RAM: {ram:.2f} GB")
            if (i+1) % 10000 == 0:
                # Plot and save chart
                plt.figure(figsize=(10, 6))
                plt.subplot(3, 1, 1)
                plt.plot(op_times, label=f'{name} op time (ms)')
                plt.ylabel('Op time (ms)')
                plt.legend(loc='upper left')
                plt.subplot(3, 1, 2)
                plt.plot(cpu_usages, label=f'{name} CPU %')
                plt.ylabel('CPU %')
                plt.legend(loc='upper left')
                plt.subplot(3, 1, 3)
                plt.plot(ram_usages, label=f'{name} RAM (GB)', color='orange')
                plt.xlabel('Batch (x1000 ops)')
                plt.ylabel('RAM (GB)')
                plt.legend(loc='upper left')
                plt.tight_layout()
                chart_path = os.path.join(PROFILES_DIR, f'{name}_stats_{(i+1)//10000:03d}.png')
                plt.savefig(chart_path)
                plt.close()
                gif_frames.append(chart_path)
        elapsed = time.time() - start
        print(f"{name} total time: {elapsed:.2f} seconds | Avg op: {sum(op_times)/len(op_times):.4f} ms")
        return op_times, cpu_usages, ram_usages

    # Benchmark standard matmul
    matmul_func = lambda a, b: torch.matmul(a, b)
    matmul_times, matmul_cpu, matmul_ram = benchmark_op("matmul", matmul_func, a_batch, b_batch)
    stats["matmul"] = matmul_times
    cpu_stats["matmul"] = matmul_cpu
    ram_stats["matmul"] = matmul_ram

    # Benchmark MultivectorOps
    mvops_func = lambda a, b: mvops.matmul_adaptive(a, b)
    mvops_times, mvops_cpu, mvops_ram = benchmark_op("mvops", mvops_func, a_batch, b_batch)
    stats["mvops"] = mvops_times
    cpu_stats["mvops"] = mvops_cpu
    ram_stats["mvops"] = mvops_ram

    # Save GIF
    if gif_frames:
        import imageio.v2 as imageio
        images = [imageio.imread(f) for f in gif_frames]
        gif_path = os.path.join(PROFILES_DIR, 'benchmark_performance.gif')
        imageio.mimsave(gif_path, images, duration=0.5)

    # Final summary chart
    plt.figure(figsize=(12, 7))
    plt.plot(stats["matmul"], label="Standard matmul (ms)", alpha=0.7)
    plt.plot(stats["mvops"], label="MultivectorOps (ms)", alpha=0.7)
    plt.xlabel("Operation #")
    plt.ylabel("Time per op (ms)")
    plt.title("Benchmark: Standard matmul vs MultivectorOps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROFILES_DIR, "benchmark_summary.png"))
    plt.close()

    print("\n--- Benchmark Summary ---")
    print(f"Standard matmul: avg {sum(stats['matmul'])/len(stats['matmul']):.4f} ms/op")
    print(f"MultivectorOps: avg {sum(stats['mvops'])/len(stats['mvops']):.4f} ms/op")
    print(f"See profiles/ for detailed charts and GIFs.")

    # --- Explicit cleanup for BitplaneEngine/MultivectorOps resources ---
    import sys
    import gc
    if hasattr(mvops, 'cache') and hasattr(mvops.cache, 'stop_worker'):
        try:
            mvops.cache.stop_worker()
        except Exception:
            pass
    gc.collect()
    sys.exit(0)