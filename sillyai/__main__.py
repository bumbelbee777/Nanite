import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
import matplotlib
import asyncio
matplotlib.use('Agg')  # Use non-interactive backend

from .model import SillyAI
from .config import ModelConfig, PrecisionLevel, Modality
from .ops import MultivectorOps
from .plugins.trainer import SillyAITrainerPlugin
from .plugins.visualizer import SillyAIVisualizerPlugin, ModelProfiler

class DynamicLearningRate:
    """Manages dynamic learning rate scheduling with reward/punishment mechanisms."""
    
    def __init__(self, initial_lr=5e-4, min_lr=1e-6, max_lr=1e-3, 
                 warmup_steps=100, reward_factor=1.1, punishment_factor=0.9,
                 patience=3, min_delta=1e-4):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.reward_factor = reward_factor
        self.punishment_factor = punishment_factor
        self.patience = patience
        self.min_delta = min_delta
        
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.step_count = 0
        self.reward_history = []
        
    def step(self, loss, optimizer):
        """Update learning rate based on loss improvement."""
        self.step_count += 1
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            self.current_lr = self.initial_lr * (self.step_count / self.warmup_steps)
            self._update_optimizer(optimizer)
            return
            
        # Check for improvement
        if loss < (self.best_loss - self.min_delta):
            # Reward: Increase learning rate
            self.current_lr = min(self.current_lr * self.reward_factor, self.max_lr)
            self.best_loss = loss
            self.no_improvement_count = 0
            self.reward_history.append(1)  # 1 for reward
        else:
            # Punishment: Decrease learning rate
            self.current_lr = max(self.current_lr * self.punishment_factor, self.min_lr)
            self.no_improvement_count += 1
            self.reward_history.append(-1)  # -1 for punishment
            
        # Update optimizer
        self._update_optimizer(optimizer)
        
    def _update_optimizer(self, optimizer):
        """Update optimizer learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr
            
    def get_reward_stats(self):
        """Get statistics about rewards and punishments."""
        if not self.reward_history:
            return {"rewards": 0, "punishments": 0, "ratio": 0}
            
        rewards = sum(1 for x in self.reward_history if x > 0)
        punishments = sum(1 for x in self.reward_history if x < 0)
        ratio = rewards / (rewards + punishments) if (rewards + punishments) > 0 else 0
        
        return {
            "rewards": rewards,
            "punishments": punishments,
            "ratio": ratio
        }

class SchrödingerDataset(Dataset):
    def __init__(self, seq_len=64, potential_type='harmonic', num_samples=1000, 
                 difficulty_level=1, max_difficulty=3):
        self.seq_len = seq_len
        self.potential_type = potential_type
        self.num_samples = num_samples
        self.difficulty_level = difficulty_level
        self.max_difficulty = max_difficulty
        self.x = np.linspace(-1, 1, seq_len)

    def __len__(self):
        return self.num_samples

    def _make_sample(self):
        if self.potential_type == 'harmonic':
            # Adjust potential strength based on difficulty
            k_min = 1.0 + (self.difficulty_level - 1) * 2.0
            k_max = 3.0 + (self.difficulty_level - 1) * 2.0
            k = np.random.uniform(k_min, k_max)
            V = 0.5 * k * self.x**2
            
            # Add complexity based on difficulty level
            if self.difficulty_level > 1:
                # Add small perturbations
                perturbation = 0.1 * (self.difficulty_level - 1) * np.sin(5 * self.x)
                V += perturbation
                
            if self.difficulty_level > 2:
                # Add multiple wells
                V += 0.2 * np.exp(-10 * (self.x - 0.5)**2) + 0.2 * np.exp(-10 * (self.x + 0.5)**2)
            
            # Generate wavefunction
            if self.difficulty_level == 1:
                # Simple ground state
                psi = np.exp(-np.sqrt(k) * self.x**2/2)
            elif self.difficulty_level == 2:
                # Mix of ground and first excited state
                psi = 0.8 * np.exp(-np.sqrt(k) * self.x**2/2) + 0.2 * self.x * np.exp(-np.sqrt(k) * self.x**2/2)
            else:
                # Complex superposition
                psi = (0.6 * np.exp(-np.sqrt(k) * self.x**2/2) + 
                      0.3 * self.x * np.exp(-np.sqrt(k) * self.x**2/2) +
                      0.1 * (2 * self.x**2 - 1) * np.exp(-np.sqrt(k) * self.x**2/2))
        else:
            # Free particle case
            V = np.zeros_like(self.x)
            if self.difficulty_level == 1:
                # Simple sine wave
                psi = np.sin(np.pi * (self.x + 1)/2)
            elif self.difficulty_level == 2:
                # Two sine waves
                psi = 0.7 * np.sin(np.pi * (self.x + 1)/2) + 0.3 * np.sin(2 * np.pi * (self.x + 1)/2)
            else:
                # Complex wave packet
                psi = (0.5 * np.sin(np.pi * (self.x + 1)/2) + 
                      0.3 * np.sin(2 * np.pi * (self.x + 1)/2) +
                      0.2 * np.sin(3 * np.pi * (self.x + 1)/2))
        
        # Normalize wavefunction
        psi = psi/np.linalg.norm(psi)
        
        # Ensure correct shapes [seq_len, 1]
        V = V.astype(np.float32)[:,None]
        psi = psi.astype(np.float32)[:,None]
        return V, psi

    def __getitem__(self, idx):
        V, psi = self._make_sample()
        return torch.tensor(V), torch.tensor(psi)

async def main():
    console = Console()
    console.print(Panel.fit(
        "[bold blue]SillyAI[/bold blue] - Quantum Wavefunction Learning",
        border_style="blue"
    ))
    
    # Initialize configuration
    config = ModelConfig(
        # Input/Output dimensions
        input_dim=1,       # Input dimension (single value per timestep)
        output_dim=1,      # Output dimension (single wavefunction value)
        d_model=64,        # Model dimension (matching sequence length)
        d_ff=128,          # Feed-forward dimension
        
        # Transformer architecture
        n_heads=4,         # Number of attention heads
        n_layers=2,        # Number of transformer layers
        max_seq_len=64,    # Maximum sequence length (matching dataset)
        dropout=0.2,       # Dropout rate
        
        # Device and optimization
        device='cpu',      # Device to run on
        precision=PrecisionLevel.TERNARY,  # Use ternary precision
        
        # Concept graph parameters
        concept_graph_size=500,  # Concept graph size
        
        # Modality support
        supported_modalities={Modality.TEXT, Modality.IMAGE},  # Support both text and image
        
        # Plugin configuration
        enabled_plugins=['trainer', 'visualizer']  # Enable core plugins
    )
    
    # Initialize ops and model
    ops = MultivectorOps()
    await ops.cache.start()  # Start the cache worker
    model = SillyAI(config, ops=ops)
    
    try:
        # Curriculum learning parameters
        num_difficulty_levels = 3
        epochs_per_level = 7
        total_epochs = num_difficulty_levels * epochs_per_level
        
        # Initialize dynamic learning rate manager
        lr_manager = DynamicLearningRate(
            initial_lr=5e-4,
            min_lr=1e-6,
            max_lr=1e-3,
            warmup_steps=100,
            reward_factor=1.1,
            punishment_factor=0.9,
            patience=3,
            min_delta=1e-4
        )
        
        # Initialize trainer plugin with improved settings
        trainer = SillyAITrainerPlugin(config)
        
        # Initialize visualizer plugin and profiler
        visualizer = SillyAIVisualizerPlugin()
        visualizer.on_init(model)
        profiler = ModelProfiler(save_dir="profiles")
        
        # Create directories for visualizations
        os.makedirs("profiles", exist_ok=True)
        os.makedirs("wavefunctions", exist_ok=True)
        
        # Load existing weights if found
        if os.path.exists('best.pt'):
            console.print("[green]💾 Found existing model checkpoint[/green]")
            try:
                # Load checkpoint
                checkpoint = torch.load('best.pt')
                
                # Check if it's a state dict or full checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                # Filter out size mismatches
                model_state_dict = model.state_dict()
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() 
                    if k in model_state_dict and v.shape == model_state_dict[k].shape
                }
                
                # Load compatible weights
                model.load_state_dict(filtered_state_dict, strict=False)
                
                # Log loaded parameters
                console.print("\n[cyan]Model State Summary:[/cyan]")
                for name, param in model.named_parameters():
                    console.print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
                    
                # Log any mismatched parameters
                missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
                if missing_keys:
                    console.print("\n[yellow]⚠️ Some parameters were not loaded due to size mismatch:[/yellow]")
                    for key in missing_keys:
                        console.print(f"  {key}: expected {model_state_dict[key].shape}, got {state_dict[key].shape if key in state_dict else 'missing'}")
            except Exception as e:
                console.print(f"[red]⚠️ Error loading model: {str(e)}[/red]")
                console.print("[yellow]⚠️ Training from scratch[/yellow]")
        else:
            console.print("[yellow]⚠️ No existing model found, training from scratch[/yellow]")

        # Train the model with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Create main task
            task = progress.add_task("[cyan]Training...", total=total_epochs)
            
            # Curriculum learning loop
            for difficulty in range(1, num_difficulty_levels + 1):
                console.print(f"\n[bold blue]Starting difficulty level {difficulty}/{num_difficulty_levels}[/bold blue]")
                
                # Create datasets for current difficulty level
                train_dataset = SchrödingerDataset(
                    seq_len=config.max_seq_len,  # Use config's sequence length
                    num_samples=2000,
                    difficulty_level=difficulty,
                    max_difficulty=num_difficulty_levels
                )
                val_dataset = SchrödingerDataset(
                    seq_len=config.max_seq_len,  # Use config's sequence length
                    num_samples=500,
                    difficulty_level=difficulty,
                    max_difficulty=num_difficulty_levels
                )
                
                # Setup trainer for current difficulty
                trainer.setup(
                    model=model,
                    ops=ops,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    batch_size=16,
                    seq_len=config.max_seq_len,  # Use config's sequence length
                    lr=lr_manager.current_lr,
                    epochs=epochs_per_level,
                    early_stop=5,
                    weight_decay=1e-4
                )
                
                # Training loop for current difficulty
                for epoch in range(epochs_per_level):
                    # Start epoch profiling
                    profiler.start_epoch()
                    epoch_start = time.time()
                    
                    # Training phase
                    model.train()
                    train_loss = await trainer.train_epoch()
                    
                    # Validation phase
                    val_loss = await trainer.validate()
                    
                    # Update learning rate based on validation loss
                    old_lr = trainer.optimizer.param_groups[0]['lr']
                    trainer.scheduler.step(val_loss)
                    new_lr = trainer.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"\n\033[93m📉 Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}\033[0m")
                    
                    # Inference test
                    test_loss = await trainer.run_inference_test()
                    
                    # Update progress
                    current_epoch = (difficulty - 1) * epochs_per_level + epoch + 1
                    progress.update(task, advance=1, 
                                  description=f"[cyan]Difficulty {difficulty}/{num_difficulty_levels} - Epoch {epoch+1}/{epochs_per_level}")
                    
                    # Record metrics
                    metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'learning_rate': new_lr,
                        'difficulty_level': difficulty
                    }
                    
                    # End epoch profiling
                    profiler.end_epoch(current_epoch, metrics)
                    
                    # Create visualizations every 5 epochs
                    if current_epoch % 5 == 0:
                        # Create resource usage animation
                        profiler.create_resource_animation(
                            f"profiles/resource_usage_epoch_{current_epoch}.gif",
                            fps=5,
                            window_size=10
                        )
                        
                        # Create static resource usage plot
                        profiler.plot_resource_usage()
                        
                        # Get and display resource summary
                        summary = profiler.get_resource_summary()
                        console.print("\n[cyan]Resource Usage Summary:[/cyan]")
                        for metric, stats in summary.items():
                            console.print(f"  {metric}:")
                            for stat, value in stats.items():
                                console.print(f"    {stat}: {value:.2f}")
                        
                        # Get and display reward statistics
                        reward_stats = lr_manager.get_reward_stats()
                        console.print("\n[cyan]Learning Progress:[/cyan]")
                        console.print(f"  Rewards: {reward_stats['rewards']}")
                        console.print(f"  Punishments: {reward_stats['punishments']}")
                        console.print(f"  Reward Ratio: {reward_stats['ratio']:.2%}")
                        
                        # Print current learning rate and difficulty
                        console.print(f"\n[cyan]Current learning rate: {new_lr:.2e}[/cyan]")
                        console.print(f"[cyan]Current difficulty level: {difficulty}/{num_difficulty_levels}[/cyan]")
            
        # Print top concepts
        model.print_concepts(top_k=10)

        # Generate and print example bytecode
        bytecode = model.generate_bytecode()
        console.print("\n[bold blue]Example bytecode from concept graph:[/bold blue]")
        for idx, (op, args) in enumerate(bytecode):
            console.print(f"  {idx:03d}: {op} {args}")
            
        # Create final resource usage visualization
        profiler.create_resource_animation(
            "profiles/final_resource_usage.gif",
            fps=5,
            window_size=20
        )
        profiler.plot_resource_usage()
        
        # Display final resource summary
        final_summary = profiler.get_resource_summary()
        console.print("\n[bold green]Final Resource Usage Summary:[/bold green]")
        for metric, stats in final_summary.items():
            console.print(f"  {metric}:")
            for stat, value in stats.items():
                console.print(f"    {stat}: {value:.2f}")
                
        # Display final learning statistics
        final_reward_stats = lr_manager.get_reward_stats()
        console.print("\n[bold green]Final Learning Statistics:[/bold green]")
        console.print(f"  Total Rewards: {final_reward_stats['rewards']}")
        console.print(f"  Total Punishments: {final_reward_stats['punishments']}")
        console.print(f"  Final Reward Ratio: {final_reward_stats['ratio']:.2%}")
        
    finally:
        await ops.cache.stop()  # Stop the cache worker

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()