import logging
import os
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse

import aiohttp
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import ModelConfig
from .configs.large import LargeModelConfig
from .configs.medium import MediumModelConfig
from .configs.small import SmallModelConfig
from .model import SillyAI
from .utils import Logger


class WikipediaCrawler:
    """Wikipedia crawler for training data collection."""

    def __init__(self, max_depth: int = 7, max_pages: int = 100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: set[str] = set()
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session."""
        await self.session.close()

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

    async def crawl_page(self, url: str) -> dict[str, str]:
        """Crawl a single Wikipedia page and return its content and links."""
        if url in self.visited_urls:
            return {}

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {}

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract content and links
                content = self._extract_text(soup)
                links = self._extract_links(soup, url)

                self.visited_urls.add(url)
                return {"url": url, "content": content, "links": links}

        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
            return {}

    async def crawl(self, start_url: str) -> list[dict[str, str]]:
        """Crawl Wikipedia pages starting from a given URL."""
        if not self._is_valid_wiki_url(start_url):
            raise ValueError("Invalid Wikipedia URL")

        pages = []
        queue = deque([(start_url, 0)])  # (url, depth)

        while queue and len(pages) < self.max_pages:
            url, depth = queue.popleft()

            if depth > self.max_depth:
                continue

            page_data = await self.crawl_page(url)
            if page_data:
                pages.append(page_data)

                # Add linked pages to queue
                for link in page_data["links"]:
                    if link not in self.visited_urls:
                        queue.append((link, depth + 1))

        return pages


class PartialSolver(SillyAI):
    """PartialSolver - A lightweight NLP chatbot and equation solver."""

    # Model size configurations
    MODEL_CONFIGS = {
        "tiny": SmallModelConfig,
        "small": SmallModelConfig,
        "medium": MediumModelConfig,
        "large": LargeModelConfig,
    }

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        max_seq_len: int = 1024,
        vocab_size: int = 32000,
        compile_mode: str = "reduce-overhead",  # 'default', 'reduce-overhead', or 'max-autotune'
    ):
        """Initialize PartialSolver.

        Args:
            model_size: Model size ('tiny', 'small', 'medium', 'large')
            device: Device to use ('cpu' or 'cuda')
            checkpoint_dir: Directory to save/load checkpoints
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            compile_mode: torch.compile() mode for maximum performance
        """
        # Initialize logger
        self.logger = Logger(
            f"partial_solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Set up model configuration
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model size: {model_size}")

        # Get the appropriate config class
        config_class = self.MODEL_CONFIGS[model_size]

        # Create config instance with overrides
        config = config_class(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            device=device,
        )

        # Initialize base model
        super().__init__(config)

        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize text modality
        self.modality_manager.add_modality(
            "text",
            {
                "w2v_path": "pretrained/GoogleNews-vectors-negative300.bin",
                "unk_token": "<UNK>",
                "pad_token": "<PAD>",
                "lowercase": True,
            },
        )

        # Initialize Wikipedia crawler
        self.crawler = WikipediaCrawler()

        # Compile everything for maximum performance
        self._compile_model(compile_mode)

        # Load latest checkpoint if available
        self.load_latest_checkpoint()

    def _compile_model(self, mode: str = "reduce-overhead"):
        """Compile the entire model for maximum performance."""
        self.logger.info(f"Compiling model with mode: {mode}")

        # Compile the main model
        self.model = torch.compile(self.model, mode=mode)

        # Compile all submodules
        for name, module in self.named_modules():
            if isinstance(module, (nn.Module, nn.ModuleList, nn.ModuleDict)):
                try:
                    setattr(self, name, torch.compile(module, mode=mode))
                except Exception as e:
                    self.logger.warning(f"Could not compile module {name}: {e}")

        # Compile key methods
        self.forward = torch.compile(self.forward, mode=mode)
        self.generate_response = torch.compile(self.generate_response, mode=mode)
        self.process_text = torch.compile(self.process_text, mode=mode)

        # Compile modality manager
        self.modality_manager = torch.compile(self.modality_manager, mode=mode)

        # Compile tokenizer
        if hasattr(self, "tokenizer"):
            self.tokenizer = torch.compile(self.tokenizer, mode=mode)

        self.logger.info("Model compilation completed")

    def load_latest_checkpoint(self):
        """Load the latest checkpoint if available."""
        if not os.path.exists(self.checkpoint_dir):
            return

        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not checkpoints:
            return

        latest = max(
            checkpoints,
            key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)),
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, latest)

        try:
            # Add ModelConfig to safe globals for PyTorch 2.6+
            torch.serialization.add_safe_globals([ModelConfig])
            # Load with weights_only=False for backward compatibility
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Check if checkpoint config matches current model
            if "config" in checkpoint:
                checkpoint_config = checkpoint["config"]
                if (
                    checkpoint_config.d_model != self.config.d_model
                    or checkpoint_config.n_heads != self.config.n_heads
                    or checkpoint_config.n_layers != self.config.n_layers
                    or checkpoint_config.d_ff != self.config.d_ff
                ):
                    self.logger.warning(
                        f"Warning: Checkpoint model size ({checkpoint_config.d_model}) "
                        f"doesn't match current model size ({self.config.d_model}). "
                        f"Initializing new model.",
                    )
                    return

            # Try to load state dict
            try:
                self.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info(f"Loaded checkpoint: {latest}")
            except Exception as e:
                self.logger.warning(
                    f"Could not load checkpoint weights due to size mismatch. "
                    f"Initializing new model. Error: {e}",
                )
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def save_checkpoint(self):
        """Save a checkpoint."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{timestamp}.pt",
        )

        try:
            # Save with weights_only=False for backward compatibility
            torch.save(
                {"model_state_dict": self.state_dict(), "config": self.config},
                checkpoint_path,
                weights_only=False,
            )
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    async def chat(self, message: str) -> str:
        """Generate a chat response using compiled operations."""
        try:
            # Process input with compiled tokenizer
            input_tensor = self.process_text(message)

            # Generate response with compiled model
            response = await self.generate_response(
                input_tensor,
                num_tokens=self.max_tokens,
                temperature=0.7,
            )

            # Convert response to text with compiled decoder
            response_text = self.tokenizer.decode(response)
            self.logger.info(f"Generated response: {response_text}")

            return response_text
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return (
                "I apologize, but I encountered an error while processing your request."
            )

    async def solve_equation(self, equation: str) -> str:
        """Solve a mathematical equation using compiled operations."""
        try:
            # Process input with compiled tokenizer
            input_tensor = self.process_text(equation)

            # Generate solution with compiled model
            solution = await self.generate_response(
                input_tensor,
                num_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more precise math
            )

            # Convert solution to text with compiled decoder
            solution_text = self.tokenizer.decode(solution)
            self.logger.info(f"Generated solution: {solution_text}")

            return solution_text
        except Exception as e:
            self.logger.error(f"Error solving equation: {e}")
            return "I apologize, but I encountered an error while solving the equation."

    def get_model_stats(self) -> dict[str, int | float]:
        """Calculate model statistics.

        Returns:
            Dictionary containing model statistics:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - non_trainable_params: Number of non-trainable parameters
            - total_layers: Total number of layers
            - total_neurons: Total number of neurons
            - avg_params_per_neuron: Average parameters per neuron
            - model_size_mb: Model size in megabytes
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        total_layers = 0
        total_neurons = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                total_layers += 1
                if hasattr(module, "out_features"):
                    total_neurons += module.out_features
                elif hasattr(module, "out_channels"):
                    total_neurons += module.out_channels

        # Calculate average parameters per neuron
        avg_params_per_neuron = total_params / total_neurons if total_neurons > 0 else 0

        # Calculate model size in MB (assuming 32-bit floats)
        model_size_mb = total_params * 4 / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "total_layers": total_layers,
            "total_neurons": total_neurons,
            "avg_params_per_neuron": avg_params_per_neuron,
            "model_size_mb": model_size_mb,
        }

    def print_model_stats(self):
        """Print model statistics in a formatted way."""
        stats = self.get_model_stats()

        print("\n" + "=" * 50)
        print("🤖 Model Statistics")
        print("=" * 50)
        print(f"📊 Total Parameters: {stats['total_params']:,}")
        print(f"🎯 Trainable Parameters: {stats['trainable_params']:,}")
        print(f"🔒 Non-trainable Parameters: {stats['non_trainable_params']:,}")
        print(f"📈 Total Layers: {stats['total_layers']}")
        print(f"🧠 Total Neurons: {stats['total_neurons']:,}")
        print(f"⚖️ Average Parameters per Neuron: {stats['avg_params_per_neuron']:.2f}")
        print(f"💾 Model Size: {stats['model_size_mb']:.2f} MB")
        print("=" * 50 + "\n")

    async def train_on_wikipedia(self, start_url: str, max_pages: int = 100):
        """Train the model on Wikipedia pages."""
        self.logger.info(f"Starting Wikipedia training from {start_url}")
        print("\n🌐 Starting Wikipedia Training")
        print(f"📚 Starting URL: {start_url}")
        print(f"📑 Max Pages: {max_pages}")
        print(f"💻 Device: {self.config.device}")
        print(f"⚡ Compilation Mode: {self.compile_mode}")
        print("\n" + "=" * 50)

        # Print model statistics before training
        self.print_model_stats()

        # Crawl Wikipedia pages
        pages = await self.crawler.crawl(start_url)

        if not pages:
            self.logger.error("No pages found to train on")
            return

        # Initialize training metrics
        total_loss = 0
        best_loss = float("inf")
        patience_counter = 0
        early_stop_patience = 5

        # Create progress bar
        pbar = tqdm(pages, desc="Training Progress", unit="pages")

        for page in pbar:
            # Process page content
            content = page["content"]
            if not content:
                continue

            # Split content into chunks
            chunks = self._split_into_chunks(content)

            for chunk in chunks:
                # Process input
                input_tensor = self.process_text(chunk)

                # Generate target response (next chunk or summary)
                target_tensor = self.process_text(self._generate_target(chunk, chunks))

                # Train step
                loss = self._train_step(input_tensor, target_tensor)
                total_loss += loss

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg_loss": f"{total_loss / (len(pages) + 1):.4f}",
                    },
                )

                # Save checkpoint if loss improved
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                    self.save_checkpoint()
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stop_patience:
                    print("\n⚠️ Early stopping triggered")
                    break

        # Print final statistics
        print("\n" + "=" * 50)
        print("✨ Training Completed")
        print("=" * 50)
        print(f"📊 Final Average Loss: {total_loss / len(pages):.4f}")
        print(f"🏆 Best Loss: {best_loss:.4f}")
        print(f"📈 Pages Trained: {len(pages)}")
        print("=" * 50 + "\n")

        self.logger.info("Wikipedia training completed")

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

    def _generate_target(self, chunk: str, all_chunks: list[str]) -> str:
        """Generate target text for training."""
        # Try to find the next chunk
        try:
            idx = all_chunks.index(chunk)
            if idx + 1 < len(all_chunks):
                return all_chunks[idx + 1]
        except ValueError:
            pass

        # If no next chunk, generate a summary
        return self._generate_summary(chunk)

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the input text."""
        # Simple extractive summarization
        sentences = text.split(".")
        if len(sentences) <= 3:
            return text

        # Take first and last sentence
        summary = sentences[0] + ". " + sentences[-1] + "."
        return summary

    def _train_step(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> float:
        """Perform a single training step with compiled operations."""
        # Forward pass with compiled model
        output = self.model(input_tensor)

        # Calculate loss with compiled operations
        loss = torch.compile(nn.MSELoss())(output, target_tensor)

        # Backward pass
        loss.backward()

        # Update weights with compiled optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
