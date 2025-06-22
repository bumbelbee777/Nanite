import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import asyncio
from colorama import init, Fore, Style

import torch
from tqdm import tqdm

from nanite.config import ModelConfig, Modality, PrecisionLevel
from nanite.configs.small import SmallModelConfig
from nanite.model import Nanite
from .partial_solver import PartialSolver
from nanite.plugins.potato_trainer import PotatoModeTrainer
from nanite.utils import Logger
import gc

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
        os.system('')  # Enable ANSI escape sequences

# Configure root logger to prevent duplicate messages
logging.getLogger().handlers = []
logger = Logger(__name__)

# Global model instance
_potato_model = None

def create_potato_model() -> Nanite:
    """Create a lightweight model configuration for edge devices."""
    global _potato_model
    
    if _potato_model is None:
        logger.info(f"{Fore.CYAN}🚀 Initializing potato mode model configuration...{Style.RESET_ALL}")
        config = ModelConfig(
            d_model=1024,
            d_ff=2048,
            mlp_dim=1024,
            n_heads=8,
            n_layers=16,
            dropout=0.3,
            max_seq_len=2048,
            device="cpu",
            precision=PrecisionLevel.TERNARY,
            concept_graph_size=4096,
            concept_decay_rate=0.1,
            supported_modalities={Modality.TEXT, Modality.IMAGE},
            enabled_plugins=["trainer", "visualizer"]
        )
        
        logger.info(f"{Fore.GREEN}ℹ️ Creating Nanite model instance...{Style.RESET_ALL}")
        _potato_model = Nanite(config)
        
        # Initialize text modality with default Word2Vec model
        logger.info(f"{Fore.BLUE}ℹ️ Adding text modality...{Style.RESET_ALL}")
        
        # Add text modality with Word2Vec configuration
        _potato_model.modality_manager.add_modality(
            "text",
            {
                "w2v_path": "pretrained/GoogleNews-vectors-negative300.bin",
                "unk_token": "<UNK>",
                "pad_token": "<PAD>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "quantize": True,
                "quantize_bits": 4,
                "use_mmap": True,
                "w2v_quantize_bits": 8,  # FP8 quantization
                "w2v_cache_size": 10000,  # Cache size
                "w2v_mmap_threshold": 50000,  # Use mmap for large vocabs
            }
        )
        
        # Initialize embeddings from Word2Vec if available
        try:
            text_tokenizer = _potato_model.modality_manager.modalities["text_tokenizer"]
            if text_tokenizer._w2v_loaded:
                logger.info(f"{Fore.GREEN}✅ Word2Vec model loaded successfully!{Style.RESET_ALL}")
                text_tokenizer._init_embeddings()  # Initialize embeddings from Word2Vec
                text_tokenizer.print_w2v_stats()  # Print Word2Vec statistics
            else:
                logger.warning(f"{Fore.YELLOW}⚠️ Word2Vec model not loaded, using random embeddings{Style.RESET_ALL}")
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}⚠️ Error initializing Word2Vec embeddings: {e}{Style.RESET_ALL}")
        
        # Verify text modality was added
        logger.info(f"{Fore.YELLOW}ℹ️ Verifying text modality initialization...{Style.RESET_ALL}")
        if "text_tokenizer" not in _potato_model.modality_manager.modalities:
            logger.error(f"{Fore.RED}❌ Text modality not found in model modalities!{Style.RESET_ALL}")
            logger.error(f"{Fore.RED}Available modalities: {list(_potato_model.modality_manager.modalities.keys())}{Style.RESET_ALL}")
            raise ValueError("Text modality not properly initialized!")
        
        logger.info(f"{Fore.GREEN}✅ Text modality successfully initialized!{Style.RESET_ALL}")
    
    return _potato_model

def load_pretrained_model(model_path: str) -> Optional[Nanite]:
    """Load a pretrained model if available."""
    try:
        if os.path.exists(model_path):
            logger.info(f"{Fore.CYAN}ℹ️ Loading pretrained model from {model_path}...{Style.RESET_ALL}")
            model = torch.load(model_path)
            logger.info(f"{Fore.GREEN}✅ Successfully loaded pretrained model!{Style.RESET_ALL}")
            return model
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}❌ Error loading pretrained model: {str(e)}{Style.RESET_ALL}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Nanite CLI")
    parser.add_argument("--potato", action="store_true", help="Enable potato mode for edge devices")
    parser.add_argument("--data-dir", type=str, default="training_data", help="Directory containing training data")
    parser.add_argument("--model-path", type=str, help="Path to save/load model")
    parser.add_argument("--w2v-path", type=str, default="pretrained/GoogleNews-vectors-negative300.bin", help="Path to Word2Vec model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model to load")
    args = parser.parse_args()

    if args.potato:
        # Check for training data directory
        if not os.path.exists(args.data_dir):
            logger.error(f"{Fore.RED}❌ Training data directory '{args.data_dir}' not found!{Style.RESET_ALL}")
            return

        # Try to load pretrained model first
        model = None
        if args.pretrained:
            model = load_pretrained_model(args.pretrained)
        
        # If no pretrained model or loading failed, try to resume from checkpoint
        if model is None and args.resume:
            # Try to load from latest checkpoint
            checkpoint_dir = Path(args.checkpoint_dir)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("potato_checkpoint_*.pt"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    logger.info(f"{Fore.CYAN}💾 Loading latest checkpoint: {latest}{Style.RESET_ALL}")
                    checkpoint = torch.load(latest)
                    model = create_potato_model()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"{Fore.GREEN}✅ Resumed from checkpoint with loss: {checkpoint['loss']:.4f}{Style.RESET_ALL}")
                else:
                    logger.warning(f"{Fore.YELLOW}⚠️ No checkpoints found, starting fresh{Style.RESET_ALL}")
                    model = create_potato_model()
            else:
                logger.warning(f"{Fore.YELLOW}⚠️ Checkpoint directory not found, starting fresh{Style.RESET_ALL}")
                model = create_potato_model()
        
        # If still no model, create a new one
        if model is None:
            logger.info(f"{Fore.CYAN}🚀 Creating new potato mode model{Style.RESET_ALL}")
            model = create_potato_model()
            
        # Set w2v path if provided
        try:
            text_modality = model.modality_manager.modalities["text_tokenizer"]
            text_modality.w2v_path = args.w2v_path
        except KeyError:
            logger.error(f"{Fore.RED}❌ text_tokenizer modality not found!{Style.RESET_ALL}")
        
        # Initialize trainer with checkpoint directory
        trainer = PotatoModeTrainer(
            model, 
            args.data_dir,
            checkpoint_dir=args.checkpoint_dir
        )
        
        try:
            # Train the model
            asyncio.run(trainer.train())
            
            # Save the final model if path provided
            if args.model_path:
                logger.info(f"{Fore.CYAN}💾 Saving final model to {args.model_path}{Style.RESET_ALL}")
                torch.save(model, args.model_path)
                gc.collect()
                
        except KeyboardInterrupt:
            logger.info(f"{Fore.YELLOW}⚠️ Training interrupted by user{Style.RESET_ALL}")
            if args.model_path:
                logger.info(f"{Fore.CYAN}💾 Saving model to {args.model_path}{Style.RESET_ALL}")
                torch.save(model, args.model_path)
                gc.collect()
    else:
        # Original CLI functionality
        solver = PartialSolver()
        solver.run()

if __name__ == "__main__":
    main()
