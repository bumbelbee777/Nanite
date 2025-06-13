import argparse
import logging
import os
import re
from datetime import datetime

from .partial_solver import PartialSolver


class PartialSolverCLI:
    """Command-line interface for PartialSolver."""

    def __init__(self):
        self.parser = self._create_parser()
        self.model: PartialSolver | None = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(description="PartialSolver CLI")

        # Model configuration
        parser.add_argument(
            "--model-size",
            type=str,
            default="tiny",
            choices=["tiny", "small", "medium", "large"],
            help="Model size configuration",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Device to run model on (cpu/cuda)",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default="checkpoints",
            help="Directory for model checkpoints",
        )
        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=512,
            help="Maximum sequence length",
        )
        parser.add_argument(
            "--vocab-size",
            type=int,
            default=32000,
            help="Vocabulary size",
        )
        parser.add_argument(
            "--compile-mode",
            type=str,
            default="reduce-overhead",
            choices=["default", "reduce-overhead", "max-autotune"],
            help="torch.compile() mode for maximum performance",
        )

        # Training configuration
        parser.add_argument(
            "--train",
            action="store_true",
            help="Train model on Wikipedia data",
        )
        parser.add_argument(
            "--start-url",
            type=str,
            help="Starting Wikipedia URL for training",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=100,
            help="Maximum number of Wikipedia pages to crawl",
        )

        # Generation configuration
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=100,
            help="Maximum tokens to generate",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Sampling temperature",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Nucleus sampling probability",
        )

        return parser

    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/partial_solver_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                ),
                logging.StreamHandler(),
            ],
        )

    async def initialize_model(self, args):
        """Initialize the model with given arguments."""
        self.model = PartialSolver(
            model_size=args.model_size,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size,
            compile_mode=args.compile_mode,
        )

    async def train_model(self, args):
        """Train the model on Wikipedia data."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        if not args.start_url:
            raise ValueError("Please provide a starting Wikipedia URL with --start-url")

        logging.info(f"Starting training with Wikipedia URL: {args.start_url}")
        logging.info(f"Maximum pages to crawl: {args.max_pages}")
        logging.info(f"Using compilation mode: {args.compile_mode}")

        await self.model.train_on_wikipedia(
            start_url=args.start_url,
            max_pages=args.max_pages,
        )
        logging.info("Training completed")

    def _is_valid_wikipedia_url(self, url: str) -> bool:
        """Check if URL is a valid Wikipedia article URL."""
        pattern = r"^https?://[a-z]{2,3}\.wikipedia\.org/wiki/[^:]+$"
        return bool(re.match(pattern, url))

    async def handle_command(self, command: str, args) -> str | None:
        """Handle special commands in chat."""
        if command.startswith("@train "):
            url = command[7:].strip()
            if not self._is_valid_wikipedia_url(url):
                return "Invalid Wikipedia URL. Please provide a valid Wikipedia article URL."

            try:
                await self.model.train_on_wikipedia(url)
                return "Training completed successfully!"
            except Exception as e:
                logging.error(f"Error training on Wikipedia: {e}")
                return f"Error during training: {e!s}"

        return None

    async def chat_loop(self, args):
        """Run interactive chat loop."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        print("\nPartialSolver Chat (type 'exit' to quit)")
        print("----------------------------------------")
        print(f"Model size: {args.model_size}")
        print(f"Device: {args.device}")
        print(f"Compilation mode: {args.compile_mode}")
        print("\nSpecial commands:")
        print("  @train <url> - Train on a Wikipedia article")
        print("  @exit - Exit the chat")
        print("----------------------------------------")

        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if user_input.lower() in ["@exit", "@getout", "@quit", "@leave"]:
                    break

                # Check for special commands
                if user_input.startswith("@"):
                    response = await self.handle_command(user_input, args)
                    if response:
                        print(f"\nPartialSolver: {response}")
                        continue

                # Check if input is an equation
                if any(op in user_input for op in ["+", "-", "*", "/", "=", "^"]):
                    response = await self.model.solve_equation(user_input)
                else:
                    response = await self.model.chat(
                        user_input,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                print(f"\nPartialSolver: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {e}")
                print("\nAn error occurred. Please try again.")

    async def run(self):
        """Run the CLI."""
        args = self.parser.parse_args()
        self._setup_logging()

        try:
            # Initialize model
            await self.initialize_model(args)

            # Train if requested
            if args.train:
                await self.train_model(args)

            # Start chat loop
            await self.chat_loop(args)

        except Exception as e:
            logging.error(f"Error running CLI: {e}")
            raise
        finally:
            # Clean up
            if self.model and hasattr(self.model, "crawler"):
                await self.model.crawler.close()
