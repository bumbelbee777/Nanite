import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from matplotlib.animation import FuncAnimation

from ..plugin import SillyPlugin

logger = logging.getLogger(__name__)


class WavefunctionVisualizer:
    """Visualizes quantum wavefunctions and their evolution during training."""

    def __init__(self, save_dir: str = "profiles"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize history
        self.history = {
            "wavefunctions": [],
            "potentials": [],
            "predictions": [],
            "energies": [],
            "probabilities": [],
        }

        # Initialize plot settings
        self.fig_size = (12, 8)
        self.dpi = 150  # Increased DPI for better quality
        self.colors = {
            "potential": "#1f77b4",  # Blue
            "wavefunction": "#2ca02c",  # Green
            "prediction": "#ff7f0e",  # Orange
            "probability": "#d62728",  # Red
            "concept": "#9467bd",  # Purple
            "relationship": "#8c564b",  # Brown
        }

    def plot_wavefunction(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        ψ: torch.Tensor,
        ψ_pred: torch.Tensor,
        epoch: int,
        save_path: str | None = None,
    ):
        """Plot wavefunction and potential.

        Args:
            x: Position tensor
            V: Potential tensor
            ψ: True wavefunction tensor
            ψ_pred: Predicted wavefunction tensor
            epoch: Current epoch number
            save_path: Optional path to save the plot. If None, uses default path in save_dir
        """
        # Store history
        self.history["wavefunctions"].append(ψ.detach().cpu().numpy())
        self.history["potentials"].append(V.detach().cpu().numpy())
        self.history["predictions"].append(ψ_pred.detach().cpu().numpy())

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot wavefunction magnitude
        if torch.is_complex(ψ):
            ψ_mag = torch.abs(ψ).cpu().numpy()
            ψ_pred_mag = torch.abs(ψ_pred).cpu().numpy()

            # Magnitude plot
            plt.subplot(2, 1, 1)
            plt.plot(
                x.cpu().numpy(),
                ψ_mag,
                color=self.colors["wavefunction"],
                label="True Magnitude",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_mag,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Magnitude",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction Magnitude |ψ(x)|", fontsize=12)
            plt.title(f"Wavefunction Magnitude (Epoch {epoch})", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)

            # Phase plot
            plt.subplot(2, 1, 2)
            ψ_phase = torch.angle(ψ).cpu().numpy()
            ψ_pred_phase = torch.angle(ψ_pred).cpu().numpy()
            plt.plot(
                x.cpu().numpy(),
                ψ_phase,
                color=self.colors["wavefunction"],
                label="True Phase",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_phase,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Phase",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction Phase (rad)", fontsize=12)
            plt.title("Wavefunction Phase", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)
            plt.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            plt.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            plt.set_ylim([-np.pi * 1.1, np.pi * 1.1])  # Add some padding
        else:
            ψ_mag = ψ.cpu().numpy()
            ψ_pred_mag = ψ_pred.cpu().numpy()

            plt.subplot(2, 1, 1)
            plt.plot(
                x.cpu().numpy(),
                ψ_mag,
                color=self.colors["wavefunction"],
                label="True Wavefunction",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_mag,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Wavefunction",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction ψ(x)", fontsize=12)
            plt.title(f"Wavefunction (Epoch {epoch})", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)

        # Save plot
        plt.tight_layout()
        if save_path is None:
            save_path = (
                self.save_dir / "wavefunctions" / f"wavefunction_epoch_{epoch}.png"
            )
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close()

    def plot_probability_density(
        self,
        x: torch.Tensor,
        ψ: torch.Tensor,
        ψ_pred: torch.Tensor,
        epoch: int,
        save: bool = True,
    ):
        """Plot probability density of wavefunction."""
        # Calculate probability densities
        if torch.is_complex(ψ):
            P_true = torch.abs(ψ) ** 2
            P_pred = torch.abs(ψ_pred) ** 2
        else:
            P_true = ψ**2
            P_pred = ψ_pred**2

        # Store in history
        self.history["probabilities"].append(P_true.detach().cpu().numpy())

        # Create plot
        plt.figure(figsize=(12, 8), dpi=150)

        # Convert tensors to numpy arrays
        x_np = x.cpu().numpy()
        P_true_np = P_true.cpu().numpy()
        P_pred_np = P_pred.cpu().numpy()

        plt.plot(
            x_np,
            P_true_np,
            color=self.colors["probability"],
            label="True",
            linewidth=2,
        )
        plt.plot(
            x_np,
            P_pred_np,
            color=self.colors["prediction"],
            linestyle="--",
            label="Predicted",
            linewidth=2,
        )
        plt.xlabel("Position (x)", fontsize=12)
        plt.ylabel("Probability Density |ψ(x)|²", fontsize=12)
        plt.title(f"Probability Density (Epoch {epoch})", fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", which="major", labelsize=10)

        # Save plot
        if save:
            save_path = self.save_dir / f"probability_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.2)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def create_animation(
        self,
        x: torch.Tensor,
        predictions_history: list[torch.Tensor],
        save_path: str,
        fps: int = 5,
    ):
        """Create animation of predicted wavefunction evolution.

        Args:
            x: Position tensor.
            predictions_history: List of predicted wavefunction tensors over time.
            save_path: Path to save the animation.
            fps: Frames per second.
        """
        if not predictions_history:
            logger.warning("No prediction history available to create animation")
            return

        # Determine if we need two subplots (magnitude and phase)
        is_complex = torch.is_complex(predictions_history[0])
        if is_complex:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, dpi=150)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.fig_size, dpi=150)

        x_np = x.cpu().numpy()

        def update(frame):
            if is_complex:
                ax1.clear()
                ax2.clear()
            else:
                ax1.clear()

            # Plot wavefunction magnitude
            ψ_pred = predictions_history[frame]
            if torch.is_complex(ψ_pred):
                ψ_pred_mag = torch.abs(ψ_pred).cpu().numpy()
                ψ_pred_phase = torch.angle(ψ_pred).cpu().numpy()
            else:
                ψ_pred_mag = ψ_pred.cpu().numpy()

            # Magnitude plot
            ax1.plot(
                x_np,
                ψ_pred_mag,
                color=self.colors["prediction"],
                label="Predicted Magnitude",
                linewidth=2,
            )
            ax1.set_xlabel("Position (x)", fontsize=12)
            ax1.set_ylabel("Wavefunction Magnitude |ψ(x)|", fontsize=12)
            ax1.set_title(
                f"Predicted Wavefunction Evolution - Magnitude (Frame {frame})",
                fontsize=14,
                pad=20,
            )
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="both", which="major", labelsize=10)

            # Phase plot for complex numbers
            if is_complex:
                ax2.plot(
                    x_np,
                    ψ_pred_phase,
                    color=self.colors["prediction"],
                    label="Predicted Phase",
                    linewidth=2,
                )
                ax2.set_xlabel("Position (x)", fontsize=12)
                ax2.set_ylabel("Wavefunction Phase (rad)", fontsize=12)
                ax2.set_title(
                    f"Predicted Wavefunction Evolution - Phase (Frame {frame})",
                    fontsize=14,
                    pad=20,
                )
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis="both", which="major", labelsize=10)
                ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                ax2.set_yticklabels(
                    [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],
                )
                ax2.set_ylim([-np.pi * 1.1, np.pi * 1.1])  # Add some padding

            plt.tight_layout(pad=3.0)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(predictions_history),
            interval=1000 // fps,
            blit=False,
        )

        # Save animation
        try:
            anim.save(save_path, writer="pillow", fps=fps)
            logger.info(f"Animation saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save animation: {e!s}")
        finally:
            plt.close()

    def plot_energy_evolution(self, save: bool = True):
        """Plot evolution of energy levels over training."""
        if not self.history["energies"]:
            logger.warning("No energy history available")
            return

        plt.figure(figsize=self.fig_size)
        energies = np.array(self.history["energies"])

        # Plot each energy level
        for i in range(energies.shape[1]):
            plt.plot(energies[:, i], label=f"E{i + 1}")

        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title("Energy Level Evolution")
        plt.legend()
        plt.grid(True)

        if save:
            save_path = self.save_dir / "energy_evolution.png"
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

        with open(path, "w") as f:
            json.dump(history_dict, f, indent=2)

    def load_history(self, path: str):
        """Load visualization history from file."""
        with open(path) as f:
            history_dict = json.load(f)

        self.history = {
            key: [np.array(arr) for arr in values]
            for key, values in history_dict.items()
        }

    def plot_concept_graph(self, concept_graph, save_path: str):
        """Plot the concept graph as a network visualization.

        Args:
            concept_graph: The ConceptGraph instance to visualize
            save_path: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.colors import LinearSegmentedColormap

            # Create a directed graph
            G = nx.DiGraph()

            # Add nodes (concepts)
            for concept in concept_graph.get_top_concepts(
                top_k=50,
            ):  # Limit to top 50 concepts
                name, energy = concept
                G.add_node(name, energy=energy)

            # Add edges (relationships)
            for concept in G.nodes():
                related = concept_graph.get_related_concepts(
                    concept,
                    top_k=5,
                )  # Top 5 relationships
                for target, weight in related:
                    if target in G.nodes():  # Only add if target is in our top concepts
                        G.add_edge(concept, target, weight=weight)

            # Create figure
            plt.figure(figsize=(20, 20), dpi=150)

            # Calculate node positions using spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)

            # Create custom colormap for node colors
            cmap = LinearSegmentedColormap.from_list(
                "energy_cmap",
                ["#ffffff", "#9467bd"],
            )

            # Draw nodes
            node_energies = [G.nodes[n]["energy"] for n in G.nodes()]
            node_colors = [cmap(e) for e in node_energies]
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=node_colors,
                node_size=1000,
                alpha=0.8,
            )

            # Draw edges with varying widths based on weight
            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_weights,
                alpha=0.4,
                edge_color=self.colors["relationship"],
            )

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

            # Add title and legend
            plt.title("Concept Graph Visualization", fontsize=16, pad=20)

            # Add colorbar for energy levels
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=min(node_energies), vmax=max(node_energies)),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Concept Energy", fontsize=12)

            # Save the plot
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close()

            logger.info(f"Concept graph visualization saved to {save_path}")

        except ImportError as e:
            logger.error(
                f"Failed to import required libraries for concept graph visualization: {e}",
            )
        except Exception as e:
            logger.error(f"Error creating concept graph visualization: {e}")


class ModelProfiler:
    """Profiles model performance and resource usage during training."""

    def __init__(self, save_dir: str = "profiles"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize metrics history
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "learning_rate": [],
            "gpu_memory": [],
            "cpu_memory": [],
            "batch_time": [],
            "epoch_time": [],
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
            "gpu_memory": "#FF6B6B",  # Coral red
            "cpu_memory": "#4ECDC4",  # Turquoise
            "batch_time": "#45B7D1",  # Sky blue
            "epoch_time": "#96CEB4",  # Sage green
        }

    def start_epoch(self):
        """Record start of epoch."""
        self.epoch_start_time = datetime.now()

    def end_epoch(self, epoch: int, metrics: dict[str, float]):
        """Record end of epoch and update metrics."""
        # Record metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        # Record epoch time
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.metrics["epoch_time"].append(epoch_time)

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
        self.metrics["batch_time"].append(batch_time)

    def _update_resource_usage(self):
        """Update resource usage statistics."""
        # CPU and memory usage
        self.metrics["cpu_memory"].append(
            self.process.memory_info().rss / 1024 / 1024,
        )  # MB

        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.metrics["gpu_memory"].append(gpu.memoryUsed)
            else:
                self.metrics["gpu_memory"].append(0)
        except:
            self.metrics["gpu_memory"].append(0)

    def _serialize_metrics(self):
        """Convert metrics to JSON-serializable format."""
        serialized = {}
        for key, values in self.metrics.items():
            # Convert any coroutines to their results
            if isinstance(values, list):
                serialized[key] = [
                    v.result() if hasattr(v, "result") else v for v in values
                ]
            else:
                serialized[key] = values
        return serialized

    def _save_metrics(self, epoch: int):
        """Save metrics to JSON file."""
        metrics = self._serialize_metrics()

        # Convert bytecode to serializable format if present
        if "bytecode" in metrics:
            metrics["bytecode"] = [
                {"opcode": op.name if hasattr(op, "name") else str(op), "args": args}
                for op, args in metrics["bytecode"]
            ]

        # Save to file
        save_path = self.save_dir / f"metrics_epoch_{epoch}.json"
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {save_path}")

    def plot_metrics(self, save_path: str):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_loss"], label="Val Loss")
        plt.plot(self.metrics["test_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Progress")

        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics["learning_rate"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")

        # Plot memory usage
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics["cpu_memory"], label="CPU Memory")
        plt.plot(self.metrics["gpu_memory"], label="GPU Memory")
        plt.xlabel("Epoch")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.title("Memory Usage")

        # Plot batch times
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics["batch_time"])
        plt.xlabel("Batch")
        plt.ylabel("Time (s)")
        plt.title("Batch Processing Time")

        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:  # Only compute stats if we have values
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                }
        return summary

    def create_resource_animation(
        self,
        save_path: str,
        fps: int = 5,
        window_size: int = 20,
    ):
        """Create animation of resource usage over time.

        Args:
            save_path: Path to save the animation
            fps: Frames per second
            window_size: Number of epochs to show in the sliding window
        """
        if not hasattr(self, "resource_history") or len(self.resource_history) < 2:
            logger.warning("Not enough resource history data to create animation")
            return

        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)

        def update(frame):
            ax1.clear()
            ax2.clear()

            # Calculate window indices
            start_idx = max(0, frame - window_size)
            end_idx = frame + 1

            # Ensure we have data to plot
            if start_idx >= end_idx or end_idx > len(self.metrics["gpu_memory"]):
                return

            x = range(start_idx, end_idx)

            # Plot memory usage
            ax1.plot(
                x,
                self.metrics["gpu_memory"][start_idx:end_idx],
                color=self.colors["gpu_memory"],
                label="GPU Memory",
                linewidth=2,
            )
            ax1.plot(
                x,
                self.metrics["cpu_memory"][start_idx:end_idx],
                color=self.colors["cpu_memory"],
                label="CPU Memory",
                linewidth=2,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Memory (MB)")
            ax1.set_title(f"Memory Usage (Epoch {frame})")
            ax1.legend()
            ax1.grid(True)

            # Plot timing metrics
            ax2.plot(
                x,
                self.metrics["batch_time"][start_idx:end_idx],
                color=self.colors["batch_time"],
                label="Batch Time",
                linewidth=2,
            )
            ax2.plot(
                x,
                self.metrics["epoch_time"][start_idx:end_idx],
                color=self.colors["epoch_time"],
                label="Epoch Time",
                linewidth=2,
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Time (s)")
            ax2.set_title("Processing Time")
            ax2.legend()
            ax2.grid(True)

            # Add current metrics as text
            metrics_text = (
                f"GPU Memory: {self.metrics['gpu_memory'][frame]:.1f} MB\n"
                f"CPU Memory: {self.metrics['cpu_memory'][frame]:.1f} MB\n"
                f"Batch Time: {self.metrics['batch_time'][frame]:.3f} s\n"
                f"Epoch Time: {self.metrics['epoch_time'][frame]:.3f} s"
            )
            fig.text(
                0.02,
                0.02,
                metrics_text,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.metrics["gpu_memory"]),
            interval=1000 // fps,
            blit=False,
        )

        # Save animation
        try:
            anim.save(save_path, writer="pillow", fps=fps)
        except Exception as e:
            logger.error(f"Failed to save animation: {e!s}")
        finally:
            plt.close()

    def plot_resource_usage(self, save: bool = True, save_path: str | None = None):
        """Plot resource usage over time.

        Args:
            save: Whether to save the plot
            save_path: Path to save the plot. If None, uses default path in save_dir
        """
        plt.figure(figsize=self.fig_size)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)

        # Plot memory usage
        ax1.plot(
            self.metrics["gpu_memory"],
            color=self.colors["gpu_memory"],
            label="GPU Memory",
            linewidth=2,
        )
        ax1.plot(
            self.metrics["cpu_memory"],
            color=self.colors["cpu_memory"],
            label="CPU Memory",
            linewidth=2,
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Memory Usage Over Time")
        ax1.legend()
        ax1.grid(True)

        # Plot timing metrics
        ax2.plot(
            self.metrics["batch_time"],
            color=self.colors["batch_time"],
            label="Batch Time",
            linewidth=2,
        )
        ax2.plot(
            self.metrics["epoch_time"],
            color=self.colors["epoch_time"],
            label="Epoch Time",
            linewidth=2,
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Processing Time Over Time")
        ax2.legend()
        ax2.grid(True)

        # Save plot
        if save:
            plt.tight_layout()
            if save_path is None:
                save_path = self.save_dir / "resource_usage.png"
            else:
                save_path = Path(save_path)
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def get_resource_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics of resource usage."""
        summary = {}
        for key in ["gpu_memory", "cpu_memory", "batch_time", "epoch_time"]:
            if self.metrics[key]:
                values = np.array(self.metrics[key])
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "latest": float(values[-1]),
                }
        return summary


class SillyAIVisualizerPlugin(SillyPlugin):
    """Plugin for visualizing model training progress and wavefunction evolution."""

    def __init__(self, save_dir: str = "profiles"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Training metrics history
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "learning_rate": [],
            "epoch_time": [],
            "memory_usage": [],
            "gpu_usage": [],
            "cpu_usage": [],
        }

        # Wavefunction history
        self.wavefunction_history = []
        self.potential_history = []
        self.prediction_history = []

        # Resource monitoring
        self.resource_history = {
            "memory": [],
            "gpu_memory": [],
            "cpu_percent": [],
            "gpu_percent": [],
        }

    def on_init(self, model):
        """Initialize visualization plugin."""
        super().on_init(model)
        logger.info(
            f"Visualizer plugin initialized with save directory: {self.save_dir}",
        )

    def on_epoch_start(self, epoch: int):
        """Record start of epoch."""
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]):
        """Record end of epoch and update visualizations."""
        # Record metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        # Record epoch time
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.metrics_history["epoch_time"].append(epoch_time)

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
        import GPUtil
        import psutil

        # CPU and memory usage
        process = psutil.Process()
        self.resource_history["memory"].append(
            process.memory_info().rss / 1024 / 1024,
        )  # MB
        self.resource_history["cpu_percent"].append(process.cpu_percent())

        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.resource_history["gpu_memory"].append(gpu.memoryUsed)
                self.resource_history["gpu_percent"].append(gpu.load * 100)
            else:
                self.resource_history["gpu_memory"].append(0)
                self.resource_history["gpu_percent"].append(0)
        except:
            self.resource_history["gpu_memory"].append(0)
            self.resource_history["gpu_percent"].append(0)

    def _save_metrics(self, epoch: int):
        """Save metrics to file."""
        metrics_file = self.save_dir / "metrics" / f"metrics_epoch_{epoch}.json"
        import json

        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _plot_metrics(self, epoch: int):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history["train_loss"], label="Train Loss")
        plt.plot(self.metrics_history["val_loss"], label="Val Loss")
        plt.plot(self.metrics_history["test_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Progress")

        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history["learning_rate"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")

        # Plot epoch time
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history["epoch_time"])
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title("Epoch Duration")

        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / "metrics" / f"metrics_epoch_{epoch}.png")
        plt.close()

    def _plot_resource_usage(self, epoch: int):
        """Plot resource usage over time."""
        plt.figure(figsize=(15, 10))

        # Plot memory usage
        plt.subplot(2, 2, 1)
        plt.plot(self.resource_history["memory"], label="RAM", color="#1f77b4")
        plt.plot(
            self.resource_history["gpu_memory"],
            label="GPU Memory",
            color="#ff7f0e",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Memory (MB)")
        plt.title("Memory Usage Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot CPU usage
        plt.subplot(2, 2, 2)
        plt.plot(
            self.resource_history["cpu_percent"],
            label="CPU Usage",
            color="#2ca02c",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Usage (%)")
        plt.title("CPU Usage Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot GPU usage
        plt.subplot(2, 2, 3)
        plt.plot(
            self.resource_history["gpu_percent"],
            label="GPU Usage",
            color="#d62728",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Usage (%)")
        plt.title("GPU Usage Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add epoch information
        plt.subplot(2, 2, 4)
        plt.text(
            0.1,
            0.5,
            f"Epoch: {epoch}\nTotal Memory: {self.resource_history['memory'][-1]:.1f} MB\nGPU Memory: {self.resource_history['gpu_memory'][-1]:.1f} MB\nCPU Usage: {self.resource_history['cpu_percent'][-1]:.1f}%\nGPU Usage: {self.resource_history['gpu_percent'][-1]:.1f}%",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.axis("off")

        # Save plot
        plt.tight_layout()
        save_path = (
            self.save_dir / "resource_usage" / f"resource_usage_epoch_{epoch}.png"
        )
        save_path.parent.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_wavefunction(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        ψ: torch.Tensor,
        ψ_pred: torch.Tensor,
        epoch: int,
        save_path: str | None = None,
    ):
        """Plot wavefunction and potential.

        Args:
            x: Position tensor
            V: Potential tensor
            ψ: True wavefunction tensor
            ψ_pred: Predicted wavefunction tensor
            epoch: Current epoch number
            save_path: Optional path to save the plot. If None, uses default path in save_dir
        """
        # Store history
        self.wavefunction_history.append(ψ.detach().cpu().numpy())
        self.potential_history.append(V.detach().cpu().numpy())
        self.prediction_history.append(ψ_pred.detach().cpu().numpy())

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot wavefunction magnitude
        if torch.is_complex(ψ):
            ψ_mag = torch.abs(ψ).cpu().numpy()
            ψ_pred_mag = torch.abs(ψ_pred).cpu().numpy()

            # Magnitude plot
            plt.subplot(2, 1, 1)
            plt.plot(
                x.cpu().numpy(),
                ψ_mag,
                color=self.colors["wavefunction"],
                label="True Magnitude",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_mag,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Magnitude",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction Magnitude |ψ(x)|", fontsize=12)
            plt.title(f"Wavefunction Magnitude (Epoch {epoch})", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)

            # Phase plot
            plt.subplot(2, 1, 2)
            ψ_phase = torch.angle(ψ).cpu().numpy()
            ψ_pred_phase = torch.angle(ψ_pred).cpu().numpy()
            plt.plot(
                x.cpu().numpy(),
                ψ_phase,
                color=self.colors["wavefunction"],
                label="True Phase",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_phase,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Phase",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction Phase (rad)", fontsize=12)
            plt.title("Wavefunction Phase", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)
            plt.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            plt.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            plt.set_ylim([-np.pi * 1.1, np.pi * 1.1])  # Add some padding
        else:
            ψ_mag = ψ.cpu().numpy()
            ψ_pred_mag = ψ_pred.cpu().numpy()

            plt.subplot(2, 1, 1)
            plt.plot(
                x.cpu().numpy(),
                ψ_mag,
                color=self.colors["wavefunction"],
                label="True Wavefunction",
                linewidth=2,
            )
            plt.plot(
                x.cpu().numpy(),
                ψ_pred_mag,
                color=self.colors["prediction"],
                linestyle="--",
                label="Predicted Wavefunction",
                linewidth=2,
            )
            plt.xlabel("Position (x)", fontsize=12)
            plt.ylabel("Wavefunction ψ(x)", fontsize=12)
            plt.title(f"Wavefunction (Epoch {epoch})", fontsize=14, pad=20)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=10)

        # Save plot
        plt.tight_layout()
        if save_path is None:
            save_path = (
                self.save_dir / "wavefunctions" / f"wavefunction_epoch_{epoch}.png"
            )
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close()

    def _create_wavefunction_animation(self, epoch: int):
        """Create animation of wavefunction evolution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        def update(frame):
            ax1.clear()
            ax2.clear()

            # Plot wavefunction
            ax1.plot(self.x, self.wavefunction_history[frame], "b-", label="True")
            ax1.plot(self.x, self.prediction_history[frame], "r--", label="Predicted")
            ax1.set_xlabel("Position (x)", fontsize=12)
            ax1.set_ylabel("Wavefunction ψ(x)", fontsize=12)
            ax1.set_title(f"Wavefunction (Frame {frame})")
            ax1.legend()

            # Plot potential
            ax2.plot(self.x, self.potential_history[frame], "k-", label="Potential")
            ax2.set_xlabel("Position (x)", fontsize=12)
            ax2.set_ylabel("Potential Energy V(x)", fontsize=12)
            ax2.set_title("Potential Energy")
            ax2.legend()

            plt.tight_layout()

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.wavefunction_history),
            interval=200,
            blit=False,
        )

        # Save animation
        anim.save(
            self.save_dir
            / "wavefunctions"
            / f"wavefunction_evolution_epoch_{epoch}.gif",
            writer="pillow",
            fps=5,
        )
        plt.close()

    def on_save(self) -> dict:
        """Save plugin state."""
        return {
            "metrics_history": self.metrics_history,
            "resource_history": self.resource_history,
            "wavefunction_history": self.wavefunction_history,
            "potential_history": self.potential_history,
            "prediction_history": self.prediction_history,
        }

    def on_load(self, state: dict):
        """Load plugin state."""
        self.metrics_history = state["metrics_history"]
        self.resource_history = state["resource_history"]
        self.wavefunction_history = state["wavefunction_history"]
        self.potential_history = state["potential_history"]
        self.prediction_history = state["prediction_history"]


def find_best_model(model_dir: str = "checkpoints") -> str | None:
    """Find the best model checkpoint."""
    model_dir = Path(model_dir)
    checkpoints = list(model_dir.glob("best_*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def load_best_model(model, model_dir: str = "checkpoints") -> bool:
    """Load the best model checkpoint if available."""
    best_path = find_best_model(model_dir)
    if best_path:
        try:
            checkpoint = torch.load(best_path)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"\033[92m✓ Loaded best model from {best_path}\033[0m")
            return True
        except Exception as e:
            print(f"\033[91m⚠️ Error loading model: {e}\033[0m")
    return False
