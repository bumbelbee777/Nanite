import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import psutil
import GPUtil
import json

from ..plugin import SillyPlugin

logger = logging.getLogger(__name__)

class WavefunctionVisualizer:
    """Visualizes quantum wavefunctions and their evolution during training."""
    
    def __init__(self, save_dir: str = "wavefunctions"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize history
        self.history = {
            'wavefunctions': [],
            'potentials': [],
            'predictions': [],
            'energies': [],
            'probabilities': []
        }
        
        # Initialize plot settings
        self.fig_size = (12, 8)
        self.dpi = 100
        self.colors = {
            'potential': 'k',
            'wavefunction': 'b',
            'prediction': 'r',
            'probability': 'g'
        }
        
    def plot_wavefunction(self, x: torch.Tensor, V: torch.Tensor, ψ: torch.Tensor, 
                         ψ_pred: torch.Tensor, epoch: int, save: bool = True):
        """Plot wavefunction and potential at a specific epoch."""
        # Store in history
        self.history['wavefunctions'].append(ψ.detach().cpu().numpy())
        self.history['potentials'].append(V.detach().cpu().numpy())
        self.history['predictions'].append(ψ_pred.detach().cpu().numpy())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        
        # Plot potential
        ax1.plot(x.cpu().numpy(), V.cpu().numpy(), 
                color=self.colors['potential'], label='Potential')
        ax1.set_xlabel('x')
        ax1.set_ylabel('V(x)')
        ax1.set_title(f'Potential Energy (Epoch {epoch})')
        ax1.legend()
        ax1.grid(True)
        
        # Plot wavefunction
        ax2.plot(x.cpu().numpy(), ψ.cpu().numpy(), 
                color=self.colors['wavefunction'], label='True')
        ax2.plot(x.cpu().numpy(), ψ_pred.cpu().numpy(), 
                color=self.colors['prediction'], linestyle='--', label='Predicted')
        ax2.set_xlabel('x')
        ax2.set_ylabel('ψ(x)')
        ax2.set_title('Wavefunction')
        ax2.legend()
        ax2.grid(True)
        
        # Save plot
        if save:
            plt.tight_layout()
            save_path = self.save_dir / f'wavefunction_epoch_{epoch}.png'
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    def plot_probability_density(self, x: torch.Tensor, ψ: torch.Tensor, 
                               ψ_pred: torch.Tensor, epoch: int, save: bool = True):
        """Plot probability density of wavefunction."""
        # Calculate probability densities
        P_true = torch.abs(ψ)**2
        P_pred = torch.abs(ψ_pred)**2
        
        # Store in history
        self.history['probabilities'].append(P_true.detach().cpu().numpy())
        
        # Create plot
        plt.figure(figsize=self.fig_size)
        plt.plot(x.cpu().numpy(), P_true.cpu().numpy(), 
                color=self.colors['probability'], label='True')
        plt.plot(x.cpu().numpy(), P_pred.cpu().numpy(), 
                color=self.colors['prediction'], linestyle='--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('|ψ(x)|²')
        plt.title(f'Probability Density (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if save:
            save_path = self.save_dir / f'probability_epoch_{epoch}.png'
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    def create_animation(self, x: torch.Tensor, save_path: str, fps: int = 5):
        """Create animation of wavefunction evolution."""
        if len(self.history['wavefunctions']) < 2:
            logger.warning("Not enough history to create animation")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Plot potential
            ax1.plot(x.cpu().numpy(), self.history['potentials'][frame], 
                    color=self.colors['potential'], label='Potential')
            ax1.set_xlabel('x')
            ax1.set_ylabel('V(x)')
            ax1.set_title(f'Potential Energy (Frame {frame})')
            ax1.legend()
            ax1.grid(True)
            
            # Plot wavefunction
            ax2.plot(x.cpu().numpy(), self.history['wavefunctions'][frame], 
                    color=self.colors['wavefunction'], label='True')
            ax2.plot(x.cpu().numpy(), self.history['predictions'][frame], 
                    color=self.colors['prediction'], linestyle='--', label='Predicted')
            ax2.set_xlabel('x')
            ax2.set_ylabel('ψ(x)')
            ax2.set_title('Wavefunction')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.history['wavefunctions']),
                           interval=1000//fps, blit=False)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        
    def plot_energy_evolution(self, save: bool = True):
        """Plot evolution of energy levels over training."""
        if not self.history['energies']:
            logger.warning("No energy history available")
            return
            
        plt.figure(figsize=self.fig_size)
        energies = np.array(self.history['energies'])
        
        # Plot each energy level
        for i in range(energies.shape[1]):
            plt.plot(energies[:, i], label=f'E{i+1}')
            
        plt.xlabel('Epoch')
        plt.ylabel('Energy')
        plt.title('Energy Level Evolution')
        plt.legend()
        plt.grid(True)
        
        if save:
            save_path = self.save_dir / 'energy_evolution.png'
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    def save_history(self, path: str):
        """Save visualization history to file."""
        history_dict = {
            key: [arr.tolist() for arr in values]
            for key, values in self.history.items()
        }
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
            
    def load_history(self, path: str):
        """Load visualization history from file."""
        with open(path, 'r') as f:
            history_dict = json.load(f)
            
        self.history = {
            key: [np.array(arr) for arr in values]
            for key, values in history_dict.items()
        }

class ModelProfiler:
    """Profiles model performance and resource usage during training."""
    
    def __init__(self, save_dir: str = "profiles"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize metrics history
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'gpu_memory': [],
            'cpu_memory': [],
            'batch_time': [],
            'epoch_time': []
        }
        
        # Initialize timing variables
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Initialize resource monitoring
        self.process = psutil.Process()
        
        # Initialize plot settings
        self.fig_size = (15, 10)
        self.dpi = 100
        self.colors = {
            'gpu_memory': '#FF6B6B',  # Coral red
            'cpu_memory': '#4ECDC4',  # Turquoise
            'batch_time': '#45B7D1',  # Sky blue
            'epoch_time': '#96CEB4'   # Sage green
        }
        
    def start_epoch(self):
        """Record start of epoch."""
        self.epoch_start_time = datetime.now()
        
    def end_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Record end of epoch and update metrics."""
        # Record metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
        # Record epoch time
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.metrics['epoch_time'].append(epoch_time)
        
        # Update resource usage
        self._update_resource_usage()
        
        # Save metrics to file
        self._save_metrics(epoch)
        
    def start_batch(self):
        """Record start of batch."""
        self.batch_start_time = datetime.now()
        
    def end_batch(self):
        """Record end of batch."""
        batch_time = (datetime.now() - self.batch_start_time).total_seconds()
        self.metrics['batch_time'].append(batch_time)
        
    def _update_resource_usage(self):
        """Update resource usage statistics."""
        # CPU and memory usage
        self.metrics['cpu_memory'].append(self.process.memory_info().rss / 1024 / 1024)  # MB
        
        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.metrics['gpu_memory'].append(gpu.memoryUsed)
            else:
                self.metrics['gpu_memory'].append(0)
        except:
            self.metrics['gpu_memory'].append(0)
            
    def _save_metrics(self, epoch: int):
        """Save metrics to file."""
        metrics_file = self.save_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def plot_metrics(self, save_path: str):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.plot(self.metrics['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # Plot memory usage
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['cpu_memory'], label='CPU Memory')
        plt.plot(self.metrics['gpu_memory'], label='GPU Memory')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.title('Memory Usage')
        
        # Plot batch times
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['batch_time'])
        plt.xlabel('Batch')
        plt.ylabel('Time (s)')
        plt.title('Batch Processing Time')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:  # Only compute stats if we have values
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
        return summary

    def create_resource_animation(self, save_path: str, fps: int = 5, window_size: int = 20):
        """Create animation of resource usage over time.
        
        Args:
            save_path: Path to save the animation
            fps: Frames per second
            window_size: Number of epochs to show in the sliding window
        """
        if len(self.metrics['gpu_memory']) < 2:
            logger.warning("Not enough resource history to create animation")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Calculate window indices
            start_idx = max(0, frame - window_size)
            end_idx = frame + 1
            x = range(start_idx, end_idx)
            
            # Plot memory usage
            ax1.plot(x, self.metrics['gpu_memory'][start_idx:end_idx], 
                    color=self.colors['gpu_memory'], label='GPU Memory', linewidth=2)
            ax1.plot(x, self.metrics['cpu_memory'][start_idx:end_idx], 
                    color=self.colors['cpu_memory'], label='CPU Memory', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Memory (MB)')
            ax1.set_title(f'Memory Usage (Epoch {frame})')
            ax1.legend()
            ax1.grid(True)
            
            # Plot timing metrics
            ax2.plot(x, self.metrics['batch_time'][start_idx:end_idx], 
                    color=self.colors['batch_time'], label='Batch Time', linewidth=2)
            ax2.plot(x, self.metrics['epoch_time'][start_idx:end_idx], 
                    color=self.colors['epoch_time'], label='Epoch Time', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time (s)')
            ax2.set_title('Processing Time')
            ax2.legend()
            ax2.grid(True)
            
            # Add current metrics as text
            metrics_text = (
                f"GPU Memory: {self.metrics['gpu_memory'][frame]:.1f} MB\n"
                f"CPU Memory: {self.metrics['cpu_memory'][frame]:.1f} MB\n"
                f"Batch Time: {self.metrics['batch_time'][frame]:.3f} s\n"
                f"Epoch Time: {self.metrics['epoch_time'][frame]:.3f} s"
            )
            fig.text(0.02, 0.02, metrics_text, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
        # Create animation
        anim = FuncAnimation(fig, update, 
                           frames=len(self.metrics['gpu_memory']),
                           interval=1000//fps, blit=False)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        
    def plot_resource_usage(self, save: bool = True):
        """Plot resource usage over time."""
        plt.figure(figsize=self.fig_size)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        
        # Plot memory usage
        ax1.plot(self.metrics['gpu_memory'], 
                color=self.colors['gpu_memory'], label='GPU Memory', linewidth=2)
        ax1.plot(self.metrics['cpu_memory'], 
                color=self.colors['cpu_memory'], label='CPU Memory', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot timing metrics
        ax2.plot(self.metrics['batch_time'], 
                color=self.colors['batch_time'], label='Batch Time', linewidth=2)
        ax2.plot(self.metrics['epoch_time'], 
                color=self.colors['epoch_time'], label='Epoch Time', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Processing Time Over Time')
        ax2.legend()
        ax2.grid(True)
        
        # Save plot
        if save:
            plt.tight_layout()
            save_path = self.save_dir / 'resource_usage.png'
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    def get_resource_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of resource usage."""
        summary = {}
        for key in ['gpu_memory', 'cpu_memory', 'batch_time', 'epoch_time']:
            if self.metrics[key]:
                values = np.array(self.metrics[key])
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1])
                }
        return summary

class SillyAIVisualizerPlugin(SillyPlugin):
    """Plugin for visualizing model training progress and wavefunction evolution."""
    
    def __init__(self, save_dir: str = "visualizations"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'memory_usage': [],
            'gpu_usage': [],
            'cpu_usage': []
        }
        
        # Wavefunction history
        self.wavefunction_history = []
        self.potential_history = []
        self.prediction_history = []
        
        # Resource monitoring
        self.resource_history = {
            'memory': [],
            'gpu_memory': [],
            'cpu_percent': [],
            'gpu_percent': []
        }
        
        # Create subdirectories
        (self.save_dir / 'metrics').mkdir(exist_ok=True)
        (self.save_dir / 'wavefunctions').mkdir(exist_ok=True)
        (self.save_dir / 'resources').mkdir(exist_ok=True)
        
    def on_init(self, model):
        """Initialize visualization plugin."""
        super().on_init(model)
        logger.info(f"Visualizer plugin initialized with save directory: {self.save_dir}")
        
    def on_epoch_start(self, epoch: int):
        """Record start of epoch."""
        self.epoch_start_time = datetime.now()
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Record end of epoch and update visualizations."""
        # Record metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                
        # Record epoch time
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # Update resource usage
        self._update_resource_usage()
        
        # Save current state
        self._save_metrics(epoch)
        self._plot_metrics(epoch)
        self._plot_resource_usage(epoch)
        
        # Create animation if we have enough history
        if len(self.wavefunction_history) > 1:
            self._create_wavefunction_animation(epoch)
            
    def _update_resource_usage(self):
        """Update resource usage statistics."""
        import psutil
        import GPUtil
        
        # CPU and memory usage
        process = psutil.Process()
        self.resource_history['memory'].append(process.memory_info().rss / 1024 / 1024)  # MB
        self.resource_history['cpu_percent'].append(process.cpu_percent())
        
        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.resource_history['gpu_memory'].append(gpu.memoryUsed)
                self.resource_history['gpu_percent'].append(gpu.load * 100)
            else:
                self.resource_history['gpu_memory'].append(0)
                self.resource_history['gpu_percent'].append(0)
        except:
            self.resource_history['gpu_memory'].append(0)
            self.resource_history['gpu_percent'].append(0)
            
    def _save_metrics(self, epoch: int):
        """Save metrics to file."""
        metrics_file = self.save_dir / 'metrics' / f'metrics_epoch_{epoch}.json'
        import json
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def _plot_metrics(self, epoch: int):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Val Loss')
        plt.plot(self.metrics_history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # Plot epoch time
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history['epoch_time'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Epoch Duration')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics' / f'metrics_epoch_{epoch}.png')
        plt.close()
        
    def _plot_resource_usage(self, epoch: int):
        """Plot resource usage over time."""
        plt.figure(figsize=(15, 10))
        
        # Plot memory usage
        plt.subplot(2, 2, 1)
        plt.plot(self.resource_history['memory'], label='RAM')
        plt.plot(self.resource_history['gpu_memory'], label='GPU Memory')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.title('Memory Usage')
        
        # Plot CPU/GPU usage
        plt.subplot(2, 2, 2)
        plt.plot(self.resource_history['cpu_percent'], label='CPU')
        plt.plot(self.resource_history['gpu_percent'], label='GPU')
        plt.xlabel('Epoch')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.title('CPU/GPU Usage')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / 'resources' / f'resources_epoch_{epoch}.png')
        plt.close()
        
    def plot_wavefunction(self, x: torch.Tensor, V: torch.Tensor, ψ: torch.Tensor, 
                         ψ_pred: torch.Tensor, epoch: int):
        """Plot wavefunction and potential."""
        # Store history
        self.wavefunction_history.append(ψ.detach().cpu().numpy())
        self.potential_history.append(V.detach().cpu().numpy())
        self.prediction_history.append(ψ_pred.detach().cpu().numpy())
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot potential
        plt.subplot(2, 1, 1)
        plt.plot(x.cpu().numpy(), V.cpu().numpy(), 'k-', label='Potential')
        plt.xlabel('x')
        plt.ylabel('V(x)')
        plt.title(f'Potential Energy (Epoch {epoch})')
        plt.legend()
        
        # Plot wavefunction
        plt.subplot(2, 1, 2)
        plt.plot(x.cpu().numpy(), ψ.cpu().numpy(), 'b-', label='True')
        plt.plot(x.cpu().numpy(), ψ_pred.cpu().numpy(), 'r--', label='Predicted')
        plt.xlabel('x')
        plt.ylabel('ψ(x)')
        plt.title('Wavefunction')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / 'wavefunctions' / f'wavefunction_epoch_{epoch}.png')
        plt.close()
        
    def _create_wavefunction_animation(self, epoch: int):
        """Create animation of wavefunction evolution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Plot potential
            ax1.plot(self.x, self.potential_history[frame], 'k-', label='Potential')
            ax1.set_xlabel('x')
            ax1.set_ylabel('V(x)')
            ax1.set_title(f'Potential Energy (Frame {frame})')
            ax1.legend()
            
            # Plot wavefunction
            ax2.plot(self.x, self.wavefunction_history[frame], 'b-', label='True')
            ax2.plot(self.x, self.prediction_history[frame], 'r--', label='Predicted')
            ax2.set_xlabel('x')
            ax2.set_ylabel('ψ(x)')
            ax2.set_title('Wavefunction')
            ax2.legend()
            
            plt.tight_layout()
            
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.wavefunction_history),
                           interval=200, blit=False)
        
        # Save animation
        anim.save(self.save_dir / 'wavefunctions' / f'wavefunction_evolution_epoch_{epoch}.gif',
                 writer='pillow', fps=5)
        plt.close()
        
    def on_save(self) -> dict:
        """Save plugin state."""
        return {
            'metrics_history': self.metrics_history,
            'resource_history': self.resource_history,
            'wavefunction_history': self.wavefunction_history,
            'potential_history': self.potential_history,
            'prediction_history': self.prediction_history
        }
        
    def on_load(self, state: dict):
        """Load plugin state."""
        self.metrics_history = state['metrics_history']
        self.resource_history = state['resource_history']
        self.wavefunction_history = state['wavefunction_history']
        self.potential_history = state['potential_history']
        self.prediction_history = state['prediction_history']

def find_best_model(model_dir: str = ".") -> Optional[str]:
    """Find the best model checkpoint."""
    model_dir = Path(model_dir)
    checkpoints = list(model_dir.glob("best_*.pt"))
    if not checkpoints:
        return None
        
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])

def load_best_model(model, model_dir: str = ".") -> bool:
    """Load the best model checkpoint if available."""
    best_path = find_best_model(model_dir)
    if best_path:
        try:
            model.load_state_dict(torch.load(best_path))
            print(f"\033[92m✓ Loaded best model from {best_path}\033[0m")
            return True
        except Exception as e:
            print(f"\033[91m⚠️ Error loading model: {e}\033[0m")
    return False 