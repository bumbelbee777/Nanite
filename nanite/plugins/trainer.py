import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin, urlparse

import aiohttp
import numpy as np
import psutil
import torch
import torch.optim as optim
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from tqdm import tqdm

from ..model import Nanite
from ..plugin import NanitePlugin
from ..utils import (
    CYAN,
    GREEN,
    RESET,
    YELLOW,
    DatasetStats,
    OptimizedDataLoader,
)
from .dynamic_learning_rate import DynamicLearningRate
from .visualizer import ModelProfiler, load_best_model

# Constants for optimization
MAX_WORKERS = min(
    psutil.cpu_count(logical=False),
    16,
)  # Limit workers for low-end devices
BATCH_SIZE = 32  # Smaller batch size for CPU training
PREFETCH_FACTOR = min(
    4,
    max(2, psutil.cpu_count(logical=False) // 2),
)  # Adaptive prefetch based on CPU cores
CHUNK_SIZE = 1000  # Number of samples per chunk
CACHE_SIZE = 1000  # Number of samples to keep in memory cache

logger = logging.getLogger(__name__)


class WikipediaDataset(Dataset):
    """Dataset for Wikipedia articles with optimized processing."""

    def __init__(
        self,
        model: Nanite,
        start_url: str,
        max_pages: int = 100,
        cache_dir="cache_chunks",
        chunk_size=CHUNK_SIZE,
    ):
        self.model = model
        self.start_url = start_url
        self.max_pages = max_pages
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize dataset stats
        self.stats = DatasetStats()

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # Initialize LRU cache for tokenized chunks
        self.token_cache = {}
        self.cache_lock = Lock()

        # Initialize prefetch queue with adaptive size
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_FACTOR)
        self.prefetch_task = None

        # Initialize compression settings
        self.compression = {
            "enabled": True,
            "method": "zstd",  # zstd is faster than gzip
            "level": 3,  # Lower level = faster compression
        }

        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()

        # Load or create cached chunks
        self.chunks = self._load_or_create_chunks()

        # Start prefetch worker
        self._start_prefetch_worker()

    def _is_valid_wiki_url(self, url: str) -> bool:
        """Check if URL is a valid Wikipedia article."""
        parsed = urlparse(url)
        return (
            parsed.netloc.endswith("wikipedia.org")
            and "/wiki/" in parsed.path
            and ":" not in parsed.path  # Exclude special pages
        )

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract Wikipedia links from BeautifulSoup object."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/wiki/"):
                full_url = urljoin(base_url, href)
                if self._is_valid_wiki_url(full_url):
                    links.append(full_url)
        return links

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text from BeautifulSoup object."""
        # Get the main content div
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            return ""

        # Remove unwanted elements
        for element in content.find_all(["sup", "div", "table"]):
            element.decompose()

        # Get all paragraphs
        paragraphs = content.find_all("p")
        text = " ".join(
            p.get_text().strip() for p in paragraphs if p.get_text().strip()
        )

        return text

    async def _crawl_page(self, url: str) -> dict[str, str]:
        """Crawl a single Wikipedia page and return its content and links."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {}

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract content and links
                content = self._extract_text(soup)
                links = self._extract_links(soup, url)

                return {"url": url, "content": content, "links": links}

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return {}

    async def _crawl_pages(self) -> list[dict[str, str]]:
        """Crawl Wikipedia pages starting from the given URL."""
        if not self._is_valid_wiki_url(self.start_url):
            raise ValueError("Invalid Wikipedia URL")

        pages = []
        visited_urls = set()
        queue = [(self.start_url, 0)]  # (url, depth)

        while queue and len(pages) < self.max_pages:
            url, depth = queue.pop(0)

            if url in visited_urls:
                continue

            page_data = await self._crawl_page(url)
            if page_data:
                pages.append(page_data)
                visited_urls.add(url)

                # Add linked pages to queue
                for link in page_data["links"]:
                    if link not in visited_urls:
                        queue.append((link, depth + 1))

        return pages

    def _load_or_create_chunks(self) -> list[dict]:
        """Load existing chunks or create new ones from Wikipedia."""
        chunk_files = sorted(self.cache_dir.glob("wikipedia_chunk_*.npz"))

        if chunk_files:
            print(f"\n{CYAN}📚 Loading cached Wikipedia chunks...{RESET}")
            chunks = []
            for f in tqdm(chunk_files, desc="Loading chunks"):
                # Use memory mapping for large files
                if f.stat().st_size > 1e9:  # 1GB
                    chunks.append(np.load(f, mmap_mode="r"))
                else:
                    chunks.append(np.load(f))
            print(f"{GREEN}✅ Loaded {len(chunks)} chunks{RESET}")
            return chunks

        print(f"\n{CYAN}🔄 Creating new Wikipedia chunks...{RESET}")
        return self._create_chunks()

    def _create_chunks(self) -> list[dict]:
        """Create chunks from Wikipedia pages with parallel processing."""
        # Crawl Wikipedia pages
        pages = asyncio.run(self._crawl_pages())

        # Process and chunk the data in parallel
        chunks = []
        current_chunk = []

        print(f"\n{CYAN}📝 Processing Wikipedia pages...{RESET}")

        # Process data in parallel batches
        batch_size = min(100, len(pages))  # Process 100 samples at a time
        for i in tqdm(range(0, len(pages), batch_size), desc="Processing data"):
            batch = pages[i : i + batch_size]

            # Process batch in parallel
            futures = []
            for item in batch:
                future = self.thread_pool.submit(self._process_sample, item)
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                sample = future.result()
                if sample:  # Only add non-empty samples
                    current_chunk.append(sample)

                    # Save chunk when it reaches chunk_size
                    if len(current_chunk) >= self.chunk_size:
                        chunk_path = (
                            self.cache_dir / f"wikipedia_chunk_{len(chunks)}.npz"
                        )
                        self._save_chunk(current_chunk, chunk_path)
                        chunks.append(current_chunk)
                        current_chunk = []

        # Save remaining samples
        if current_chunk:
            chunk_path = self.cache_dir / f"wikipedia_chunk_{len(chunks)}.npz"
            self._save_chunk(current_chunk, chunk_path)
            chunks.append(current_chunk)

        print(f"{GREEN}✅ Created {len(chunks)} chunks{RESET}")
        return chunks

    def _process_sample(self, item: dict) -> dict | None:
        """Process a single Wikipedia page with optimized tokenization."""
        content = item["content"]
        if not content:
            return None

        # Split content into chunks
        chunks = self._split_into_chunks(content)
        if not chunks:
            return None

        # Process through model's tokenizer with memory optimization
        with torch.no_grad():  # Disable gradient tracking
            input_tokens = self.model.process_text(chunks[0])
            target_tokens = self.model.process_text(
                chunks[1] if len(chunks) > 1 else chunks[0],
            )

            # Convert to numpy for storage
            input_tokens = input_tokens.cpu().numpy()
            target_tokens = target_tokens.cpu().numpy()

        return {
            "input": chunks[0],
            "target": chunks[1] if len(chunks) > 1 else chunks[0],
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "hash": hashlib.md5(content.encode()).hexdigest(),
        }

    def _split_into_chunks(self, text: str, chunk_size: int = 512) -> list[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _save_chunk(self, chunk: list[dict], path: Path):
        """Save chunk with compression."""
        # Convert chunk to numpy arrays
        inputs = np.array([s["input"] for s in chunk])
        targets = np.array([s["target"] for s in chunk])
        input_tokens = np.stack([s["input_tokens"] for s in chunk])
        target_tokens = np.stack([s["target_tokens"] for s in chunk])
        hashes = np.array([s["hash"] for s in chunk])

        # Save with compression
        np.savez_compressed(
            path,
            inputs=inputs,
            targets=targets,
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            hashes=hashes,
        )

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
                logger.error(f"Error in prefetch worker: {e}")
                await asyncio.sleep(1)

    def _predict_next_chunk(self) -> int:
        """Predict next chunk to load based on access patterns."""
        with self.cache_lock:
            if not self.token_cache:
                return 0
            # Return least recently used chunk
            return min(self.token_cache.items(), key=lambda x: x[1][1])[0]

    async def _load_chunk_async(self, chunk_idx: int) -> np.ndarray | None:
        """Load chunk asynchronously with memory mapping."""
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
            logger.error(f"Error loading chunk {chunk_idx}: {e}")
            return None

    def _deduplicate_data(self, data: np.ndarray) -> np.ndarray:
        """Remove duplicate samples using hashing."""
        unique_hashes = set()
        unique_indices = []

        for i, sample in enumerate(data):
            if sample["hash"] not in unique_hashes:
                unique_hashes.add(sample["hash"])
                unique_indices.append(i)

        return data[unique_indices]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset with optimized loading."""
        start_time = time.time()

        # Get chunk index and sample index
        chunk_idx = idx // self.chunk_size
        sample_idx = idx % self.chunk_size

        # Try to get from cache first
        with self.cache_lock:
            if chunk_idx in self.token_cache:
                chunk_data, _ = self.token_cache[chunk_idx]
                self.token_cache[chunk_idx] = (chunk_data, time.time())
                return chunk_data[sample_idx]

        # Load chunk
        chunk_data = self.chunks[chunk_idx]

        # Update cache
        with self.cache_lock:
            # Remove oldest entry if cache is full
            if len(self.token_cache) >= CACHE_SIZE:
                oldest_idx = min(self.token_cache.items(), key=lambda x: x[1][1])[0]
                del self.token_cache[oldest_idx]

            self.token_cache[chunk_idx] = (chunk_data, time.time())

        # Update stats
        self.stats.update_load_time(time.time() - start_time)

        return chunk_data[sample_idx]

    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(len(chunk) for chunk in self.chunks)

    def print_stats(self):
        """Print dataset statistics."""
        self.stats.print_stats()

    async def close(self):
        """Close resources."""
        if self.session:
            await self.session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "session"):
            asyncio.create_task(self.close())


class NaniteTrainerPlugin(NanitePlugin):
    """Plugin for training Nanite models."""

    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.ops = None
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = 32
        self.seq_len = 64
        self.lr = 1e-3
        self.epochs = 10
        self.early_stop = 5
        self.weight_decay = 0.0
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_metrics = []
        self.resource_metrics = []
        self.profiler = ModelProfiler()

    def setup(
        self,
        model,
        ops,
        train_dataset=None,
        val_dataset=None,
        batch_size=32,
        seq_len=64,
        lr=1e-3,
        epochs=10,
        early_stop=5,
        weight_decay=0.0,
    ):
        """Setup the trainer plugin."""
        self.model = model
        self.ops = ops
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.early_stop = early_stop
        self.weight_decay = weight_decay

        # Initialize optimizer with dynamic learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Initialize learning rate scheduler
        self.scheduler = DynamicLearningRate(
            self.optimizer,
            warmup_steps=1000,
            d_model=self.model.config.d_model,
        )

        # Initialize profiler
        self.profiler.setup(self.model)

    async def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Create data loader with optimized settings
        train_loader = OptimizedDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=MAX_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

        # Training loop
        for batch in tqdm(train_loader, desc="Training"):
            # Get input and target tensors
            input_tensor, target_tensor = batch

            # Move to device
            input_tensor = input_tensor.to(self.model.config.device)
            target_tensor = target_tensor.to(self.model.config.device)

            # Forward pass
            output = self.model(input_tensor)

            # Calculate loss
            loss = self.ops.complex_loss(output, target_tensor)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update profiler
            self.profiler.update()

        return total_loss / num_batches

    async def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Create data loader with optimized settings
        val_loader = OptimizedDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=MAX_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

        # Validation loop
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get input and target tensors
                input_tensor, target_tensor = batch

                # Move to device
                input_tensor = input_tensor.to(self.model.config.device)
                target_tensor = target_tensor.to(self.model.config.device)

                # Forward pass
                output = self.model(input_tensor)

                # Calculate loss
                loss = self.ops.complex_loss(output, target_tensor)

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, epochs=None):
        """Train the model."""
        if epochs is not None:
            self.epochs = epochs

        print(f"\n{CYAN}🚀 Starting Training{RESET}")
        print(f"📝 Epochs: {self.epochs}")
        print(f"💻 Device: {self.model.config.device}")
        print(f"⚡ Batch Size: {self.batch_size}")
        print(f"📚 Learning Rate: {self.lr}")
        print("\n" + "=" * 50)

        # Training loop
        for epoch in range(self.epochs):
            print(f"\n{CYAN}Epoch {epoch + 1}/{self.epochs}{RESET}")

            # Train epoch
            train_loss = asyncio.run(self.train_epoch())

            # Validate
            val_loss = asyncio.run(self.validate())

            # Update metrics
            self.training_metrics.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss},
            )

            # Update resource metrics
            self.resource_metrics.append(self.profiler.get_metrics())

            # Print progress
            print(f"\n{GREEN}✅ Epoch {epoch + 1} completed{RESET}")
            print(f"📊 Train Loss: {train_loss:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}")

            # Check early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop:
                    print(f"\n{YELLOW}⚠️ Early stopping triggered{RESET}")
                    break

        # Print final statistics
        print("\n" + "=" * 50)
        print(f"{GREEN}✨ Training Completed{RESET}")
        print("=" * 50)
        print(f"📊 Best Val Loss: {self.best_val_loss:.4f}")
        print(f"📈 Total Epochs: {epoch + 1}")
        print("=" * 50 + "\n")

        # Plot metrics
        self.plot_training_metrics()
        self.plot_resource_usage()

    def collate_fn(self, batch):
        """Collate function for data loader."""
        input_tensors = [item[0] for item in batch]
        target_tensors = [item[1] for item in batch]

        # Pad sequences
        input_tensor = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True)
        target_tensor = torch.nn.utils.rnn.pad_sequence(
            target_tensors,
            batch_first=True,
        )

        return input_tensor, target_tensor

    def load_best_model(self):
        """Load the best model checkpoint."""
        load_best_model(self.model)

    def get_model_stats(self):
        """Get model statistics."""
        return self.model.get_model_stats()

    def print_model_stats(self):
        """Print model statistics."""
        self.model.print_model_stats()

    def plot_training_metrics(self):
        """Plot training metrics."""
        self.profiler.plot_metrics(self.training_metrics)

    def plot_resource_usage(self):
        """Plot resource usage metrics."""
        self.profiler.plot_resource_usage(self.resource_metrics)

    def create_resource_animation(self):
        """Create resource usage animation."""
        self.profiler.create_animation(self.resource_metrics)

    def on_init(self, model):
        """Called when plugin is initialized."""
        self.model = model

    def on_enable(self):
        """Called when plugin is enabled."""

    def on_disable(self):
        """Called when plugin is disabled."""

    def on_epoch_start(self, epoch):
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch, metrics):
        """Called at the end of each epoch."""

    def train_on_wikipedia(
        self,
        start_url: str,
        max_pages: int = 100,
        epochs: int = None,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """Train the model on Wikipedia pages."""
        # Create Wikipedia dataset
        self.train_dataset = WikipediaDataset(
            self.model,
            start_url=start_url,
            max_pages=max_pages,
        )

        # Setup trainer
        self.setup(
            self.model,
            self.ops,
            train_dataset=self.train_dataset,
            batch_size=batch_size,
            lr=lr,
        )

        # Start training
        self.train(epochs)

        # Cleanup
        asyncio.run(self.train_dataset.close())
