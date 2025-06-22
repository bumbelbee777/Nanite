"""A lightweight trainer for edge devices that processes text files sequentially."""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List, Optional, Dict
import asyncio
from datetime import datetime, timedelta
import time
import sys
from colorama import init, Fore, Style

import numpy as np
import torch
from numba import jit, prange
from tqdm import tqdm

from ..model import Nanite
from ..ops import TensorCache, MixedPrecisionRouter, Quantizer
from ..utils import Logger, OptimizedDataLoader, RingBuffer

# Initialize colorama with Windows support
init(autoreset=True)

# Configure Windows console for UTF-8
if sys.platform == 'win32':
    try:
        # Enable virtual terminal processing
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        
        # Set console output encoding to UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # Fallback if virtual terminal processing fails
        import os
        os.system('')  # Enable ANSI escape sequences

logger = Logger(__name__)

@dataclass
class TrainingStats:
    """Training statistics tracking."""
    start_time: datetime
    total_sentences: int = 0
    total_loss: float = 0.0
    best_loss: float = float('inf')
    current_epoch: int = 0
    sentences_this_epoch: int = 0
    epoch_loss: float = 0.0
    last_checkpoint_time: datetime = None
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed training time."""
        return datetime.now() - self.start_time
    
    @property
    def avg_loss(self) -> float:
        """Get average loss across all sentences."""
        if self.total_sentences == 0:
            return 0.0
        return self.total_loss / self.total_sentences
    
    @property
    def epoch_avg_loss(self) -> float:
        """Get average loss for current epoch."""
        if self.sentences_this_epoch == 0:
            return 0.0
        return self.epoch_loss / self.sentences_this_epoch
    
    def update(self, loss: float):
        """Update stats with new loss value."""
        self.total_sentences += 1
        self.sentences_this_epoch += 1
        self.total_loss += loss
        self.epoch_loss += loss
        if loss < self.best_loss:
            self.best_loss = loss
    
    def next_epoch(self):
        """Move to next epoch."""
        self.current_epoch += 1
        self.sentences_this_epoch = 0
        self.epoch_loss = 0.0
    
    def format_stats(self) -> str:
        """Format current stats for display with colors and emojis."""
        # Format loss values with proper handling of edge cases
        avg_loss_str = f"{self.avg_loss:.4f}" if self.total_sentences > 0 else "0.0000"
        epoch_loss_str = f"{self.epoch_avg_loss:.4f}" if self.sentences_this_epoch > 0 else "0.0000"
        best_loss_str = f"{self.best_loss:.4f}" if self.best_loss != float('inf') else "inf"
        
        # Format elapsed time to be more readable
        elapsed = self.elapsed_time
        time_str = f"{elapsed.seconds // 3600:02d}:{(elapsed.seconds % 3600) // 60:02d}:{elapsed.seconds % 60:02d}"
        
        return (
            f"{Fore.CYAN}📚 Epoch: {self.current_epoch} | "
            f"{Fore.GREEN}⏳ Sentences: {self.total_sentences} | "
            f"{Fore.YELLOW}📉 Loss: {avg_loss_str} | "
            f"{Fore.MAGENTA}Epoch Loss: {epoch_loss_str} | "
            f"{Fore.RED}🏆 Best Loss: {best_loss_str} | "
            f"{Fore.BLUE}⏱️ Time: {time_str}{Style.RESET_ALL}"
        )

@dataclass
class TextChunk:
    """Memory-efficient text chunk with preprocessed data."""
    text: str
    sentences: List[str]
    vectors: Optional[torch.Tensor] = None
    processed: bool = False

@dataclass
class BatchStats:
    """Statistics for batch processing."""
    batch_size: int
    processed_items: int = 0
    total_loss: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_loss(self) -> float:
        """Get average loss for the batch."""
        if self.processed_items == 0:
            return 0.0
        return self.total_loss / self.processed_items
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed processing time."""
        return datetime.now() - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Get processing speed in items per second."""
        elapsed = self.elapsed_time.total_seconds()
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed
    
    def format_stats(self) -> str:
        """Format batch statistics for display."""
        return (
            f"{Fore.CYAN}📦 Batch Progress: {self.processed_items}/{self.batch_size} | "
            f"{Fore.YELLOW}📉 Loss: {self.avg_loss:.4f} | "
            f"{Fore.GREEN}⚡ Speed: {self.items_per_second:.1f} items/s | "
            f"{Fore.BLUE}⏱️ Time: {self.elapsed_time.total_seconds():.1f}s{Style.RESET_ALL}"
        )

class OptimizedTextProcessor:
    """Optimized text processing with Numba acceleration."""
    
    def __init__(self, buffer_size: int = 1024):  # Reduced buffer size
        self.word_pattern = re.compile(r'\b\w+\b')
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.buffer = RingBuffer(buffer_size)
        self.cache = TensorCache(max_bytes=16384 * 1024)  # 500KB cache size
        self.precision_router = MixedPrecisionRouter()
        self.quantizer = Quantizer(qmin=-64, qmax=63)  # Reduced precision
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _process_words_numba(words: np.ndarray) -> np.ndarray:
        """Numba-accelerated word processing."""
        processed = np.empty_like(words, dtype=np.str_)
        for i in prange(len(words)):
            # Skip empty strings
            if len(words[i]) == 0:
                processed[i] = ""
                continue
                
            # Simple lowercase conversion without ord()
            word = words[i].lower()
            processed[i] = word
        return processed

    def _process_words(self, words: List[str]) -> List[str]:
        """Process words with error handling."""
        try:
            # Process words directly in Python
            return [w.lower() for w in words]
        except Exception as e:
            logger.error(f"Error processing words: {str(e)}")
            return [w.lower() for w in words]  # Fallback to simple Python implementation

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into sentences with error handling."""
        try:
            # Split into sentences
            sentences = self.sentence_pattern.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Process each sentence
            processed_sentences = []
            for sentence in sentences:
                # Extract words
                words = self.word_pattern.findall(sentence)
                if not words:
                    continue
                    
                # Process words
                processed_words = self._process_words(words)
                processed_sentences.append(" ".join(processed_words))
            
            return processed_sentences
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return []

class PotatoModeTrainer:
    """A lightweight trainer for edge devices that processes text files sequentially."""
    
    def __init__(
        self,
        model: Nanite,
        data_dir: str,
        buffer_size: int = 1024,
        num_workers: int = 1,
        chunk_size: int = 256,
        checkpoint_dir: str = "checkpoints",
        max_epochs: int = 10,
        early_stopping_patience: int = 3,
        oneshot_mode: bool = False,
        batch_size: int = 32,  # Added batch size parameter
    ):
        self.model = model
        self.data_dir = Path(data_dir)
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.oneshot_mode = oneshot_mode
        self.batch_size = batch_size
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.text_processor = OptimizedTextProcessor(buffer_size)
        self.stats = TrainingStats(start_time=datetime.now())
        self.batch_stats = BatchStats(batch_size=batch_size)
        self.lock = Lock()
        
        # Training state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        
        logger.info("🚀 Starting potato mode training...")
        if self.oneshot_mode:
            logger.info("⚡ Oneshot mode enabled - performing quick validation run")
            self.max_epochs = 1
            self.chunk_size = min(64, batch_size)  # Ensure chunk size doesn't exceed batch size
            self.buffer_size = 256
    
    def _save_checkpoint(self, loss: float, is_final: bool = False) -> None:
        """Save model checkpoint with error handling."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'epoch': self.stats.current_epoch,
                'stats': self.stats,
            }, checkpoint_path)
            
            # Update last checkpoint time
            self.stats.last_checkpoint_time = datetime.now()
            
            # Log checkpoint
            if is_final:
                logger.info(f"💾 Saved final checkpoint to {checkpoint_path}")
            else:
                logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {str(e)}")
    
    def _load_and_preprocess_file(self, file_path: Path) -> List[TextChunk]:
        """Load and preprocess a text file with error handling."""
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Preprocess text
            sentences = self.text_processor.preprocess_text(text)
            print(f"[DEBUG] File {file_path.name}: {len(sentences)} sentences after preprocessing.")
            # Create chunks
            chunks = []
            for i in range(0, len(sentences), self.chunk_size):
                chunk_sentences = sentences[i:i + self.chunk_size]
                chunks.append(TextChunk(
                    text=text,
                    sentences=chunk_sentences,
                    processed=False
                ))
            print(f"[DEBUG] File {file_path.name}: {len(chunks)} chunks created.")
            return chunks
        except Exception as e:
            logger.error(f"❌ Error processing file {file_path}: {str(e)}")
            return []
    
    async def _process_batch(self, sentences: List[str]) -> None:
        """Process a batch of sentences with detailed statistics."""
        batch_start = datetime.now()
        batch_losses = []
        print(f"[DEBUG] Processing batch with {len(sentences)} sentences: {sentences}")
        try:
            # Process each sentence in the batch
            for sentence in sentences:
                if self.should_stop:
                    break
                try:
                    # Forward pass
                    loss = await self.model.train_step(sentence)
                    print(f"[DEBUG] train_step loss for sentence: {loss}")
                    # Validate loss
                    if not isinstance(loss, (int, float)) or np.isnan(loss) or np.isinf(loss):
                        print(f"[DEBUG] Skipping invalid loss: {loss}")
                        logger.error(f"❌ Invalid loss value: {loss}")
                        continue
                    batch_losses.append(loss)
                    # Update batch stats
                    with self.lock:
                        self.batch_stats.processed_items += 1
                        self.batch_stats.total_loss += loss
                        self.stats.update(loss)
                    # Log batch progress
                    if self.batch_stats.processed_items % 10 == 0:  # Log every 10 items
                        logger.info(self.batch_stats.format_stats())
                except Exception as e:
                    logger.error(f"❌ Error processing sentence: {str(e)}")
                    continue
            # Log batch completion
            batch_time = (datetime.now() - batch_start).total_seconds()
            logger.info(
                f"✅ Batch completed in {batch_time:.2f}s | "
                f"Avg Loss: {np.mean(batch_losses):.4f}" if batch_losses else "Avg Loss: N/A" +
                f" | Speed: {len(sentences)/batch_time:.1f} items/s"
            )
        except Exception as e:
            logger.error(f"❌ Error processing batch: {str(e)}")
            raise

    async def _process_chunk(self, chunk: TextChunk) -> None:
        """Process a text chunk with batching support."""
        try:
            if chunk.processed:
                return
            
            # Split sentences into batches
            for i in range(0, len(chunk.sentences), self.batch_size):
                if self.should_stop:
                    break
                    
                batch = chunk.sentences[i:i + self.batch_size]
                await self._process_batch(batch)
                
                # Check for early stopping
                if self.stats.avg_loss < self.best_loss:
                    self.best_loss = self.stats.avg_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        self.should_stop = True
                        logger.warning("⚠️ Early stopping triggered")
                        return
            
            chunk.processed = True
            
        except Exception as e:
            logger.error(f"❌ Error processing chunk: {str(e)}")

    async def train(self) -> None:
        """Train the model with enhanced oneshot mode support."""
        try:
            text_files = list(self.data_dir.glob("*.txt"))
            if not text_files:
                logger.error("❌ No text files found in data directory")
                return
            
            logger.info(f"📄 Found {len(text_files)} text files to process")
            
            if self.oneshot_mode:
                text_files = text_files[:1]
                logger.info("⚡ Oneshot mode: Using first file only for validation")
            
            for epoch in range(self.max_epochs):
                if self.should_stop:
                    break
                
                self.stats.next_epoch()
                logger.info(f"📚 Starting epoch {epoch + 1}/{self.max_epochs}")
                
                for file_path in text_files:
                    if self.should_stop:
                        break
                    
                    logger.info(f"📄 Processing {file_path.name}")
                    
                    chunks = self._load_and_preprocess_file(file_path)
                    
                    if self.oneshot_mode:
                        chunks = chunks[:1]
                        logger.info("⚡ Oneshot mode: Processing first chunk only")
                    
                    for chunk in chunks:
                        if self.should_stop:
                            break
                            
                        await self._process_chunk(chunk)
                        logger.info(self.stats.format_stats())
                        
                        if self.oneshot_mode:
                            try:
                                logger.info("⚡ Oneshot mode: Testing response generation...")
                                test_input = "Hello, how are you?"
                                input_tensor = await self.model.process_text(test_input)
                                response = await self.model.generate_response(
                                    input_tensor,
                                    num_tokens=10,
                                    temperature=0.7
                                )
                                logger.info(f"✅ Test response: {response}")
                            except Exception as e:
                                logger.error(f"❌ Response generation test failed: {str(e)}")
                                raise
                    
                    self._save_checkpoint(self.stats.avg_loss)
                
                self._save_checkpoint(self.stats.avg_loss, is_final=True)
                
                if self.oneshot_mode:
                    logger.info("✨ Oneshot validation completed successfully!")
                    break
            
            logger.info("✨ Training complete!")
                
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            raise 