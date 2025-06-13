from collections import deque

import numpy as np
import torch


class DynamicLearningRate:
    """Manages dynamic learning rate scheduling with advanced heuristics for optimization."""

    def __init__(
        self,
        initial_lr: float = 5e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        warmup_steps: int = 100,
        reward_factor: float = 1.1,
        punishment_factor: float = 0.9,
        patience: int = 3,
        min_delta: float = 1e-4,
        momentum: float = 0.9,
        cycle_length: int = 1000,
        cycle_momentum: bool = True,
        history_size: int = 100,
        plateau_threshold: float = 0.1,
        adaptive_threshold: bool = True,
    ):
        """Initialize the dynamic learning rate manager.

        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            warmup_steps: Number of warmup steps
            reward_factor: Factor to increase learning rate on improvement
            punishment_factor: Factor to decrease learning rate on no improvement
            patience: Number of steps to wait before reducing learning rate
            min_delta: Minimum change in loss to be considered improvement
            momentum: Momentum factor for learning rate changes
            cycle_length: Length of learning rate cycles
            cycle_momentum: Whether to use cyclical momentum
            history_size: Size of loss history to keep
            plateau_threshold: Threshold for plateau detection
            adaptive_threshold: Whether to use adaptive thresholds
        """
        # Basic parameters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.reward_factor = reward_factor
        self.punishment_factor = punishment_factor
        self.patience = patience
        self.min_delta = min_delta

        # Advanced parameters
        self.momentum = momentum
        self.cycle_length = cycle_length
        self.cycle_momentum = cycle_momentum
        self.history_size = history_size
        self.plateau_threshold = plateau_threshold
        self.adaptive_threshold = adaptive_threshold

        # State variables
        self.current_lr = initial_lr
        self.best_loss = float("inf")
        self.no_improvement_count = 0
        self.step_count = 0
        self.reward_history = []
        self.rewards = 0
        self.punishments = 0
        self.loss_history = deque(maxlen=history_size)
        self.lr_history = deque(maxlen=history_size)
        self.momentum_buffer = 0.0
        self.cycle_position = 0
        self.plateau_detected = False
        self.plateau_count = 0

    def _detect_plateau(self) -> bool:
        """Detect if loss has plateaued using recent history."""
        if len(self.loss_history) < 10:
            return False

        recent_losses = list(self.loss_history)[-10:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        # Check if loss variation is small relative to mean
        if std_loss / (mean_loss + 1e-8) < self.plateau_threshold:
            return True
        return False

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on loss history."""
        if len(self.loss_history) < 5:
            return self.min_delta

        recent_losses = list(self.loss_history)[-5:]
        mean_loss = np.mean(recent_losses)
        return max(self.min_delta, mean_loss * 0.01)  # 1% of mean loss

    def _update_cycle_position(self):
        """Update position in learning rate cycle."""
        self.cycle_position = (self.cycle_position + 1) % self.cycle_length

    def _get_cyclical_lr(self) -> float:
        """Calculate cyclical learning rate."""
        cycle_progress = self.cycle_position / self.cycle_length
        if cycle_progress < 0.5:
            # Increasing phase
            return self.min_lr + (self.max_lr - self.min_lr) * (2 * cycle_progress)
        else:
            # Decreasing phase
            return self.max_lr - (self.max_lr - self.min_lr) * (
                2 * (cycle_progress - 0.5)
            )

    def _update_momentum(self, lr_change: float):
        """Update momentum buffer with learning rate change."""
        self.momentum_buffer = (
            self.momentum * self.momentum_buffer + (1 - self.momentum) * lr_change
        )

    def step(self, loss: float, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """Update learning rate based on loss improvement.

        Args:
            loss: Current loss value
            optimizer: Optimizer to update

        Returns:
            Dictionary containing current learning rate and statistics
        """
        self.step_count += 1
        self.loss_history.append(loss)
        self.lr_history.append(self.current_lr)

        # Warmup phase
        if self.step_count <= self.warmup_steps:
            self.current_lr = self.initial_lr * (self.step_count / self.warmup_steps)
            self._update_optimizer(optimizer)
            return self._get_stats()

        # Detect plateau
        self.plateau_detected = self._detect_plateau()
        if self.plateau_detected:
            self.plateau_count += 1
            if self.plateau_count >= self.patience:
                # Reset learning rate and momentum on persistent plateau
                self.current_lr = self.initial_lr
                self.momentum_buffer = 0.0
                self.plateau_count = 0
                self._update_optimizer(optimizer)
                return self._get_stats()

        # Calculate adaptive threshold if enabled
        threshold = (
            self._calculate_adaptive_threshold()
            if self.adaptive_threshold
            else self.min_delta
        )

        # Check for improvement
        if loss < (self.best_loss - threshold):
            # Reward: Increase learning rate with momentum
            lr_change = self.current_lr * (self.reward_factor - 1)
            self._update_momentum(lr_change)
            self.current_lr = min(
                self.current_lr * self.reward_factor + self.momentum_buffer,
                self.max_lr,
            )
            self.best_loss = loss
            self.no_improvement_count = 0
            self.rewards += 1
            self.reward_history.append(1)
        else:
            # Punishment: Decrease learning rate with momentum
            lr_change = self.current_lr * (1 - self.punishment_factor)
            self._update_momentum(-lr_change)
            self.current_lr = max(
                self.current_lr * self.punishment_factor - self.momentum_buffer,
                self.min_lr,
            )
            self.no_improvement_count += 1
            self.punishments += 1
            self.reward_history.append(0)

        # Update cyclical learning rate if enabled
        if self.cycle_length > 0:
            self._update_cycle_position()
            cyclical_lr = self._get_cyclical_lr()
            # Blend current learning rate with cyclical learning rate
            self.current_lr = 0.7 * self.current_lr + 0.3 * cyclical_lr

        # Update optimizer
        self._update_optimizer(optimizer)

        return self._get_stats()

    def _update_optimizer(self, optimizer: torch.optim.Optimizer):
        """Update optimizer learning rate."""
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def _get_stats(self) -> dict[str, float]:
        """Get current statistics about learning rate and rewards."""
        total = self.rewards + self.punishments
        ratio = self.rewards / total if total > 0 else 0

        return {
            "current_lr": self.current_lr,
            "rewards": self.rewards,
            "punishments": self.punishments,
            "ratio": ratio,
            "momentum": self.momentum_buffer,
            "plateau_detected": self.plateau_detected,
            "best_loss": self.best_loss,
        }

    def get_reward_stats(self) -> dict[str, float]:
        """Get statistics about rewards and punishments."""
        total = self.rewards + self.punishments
        ratio = self.rewards / total if total > 0 else 0

        # Calculate recent reward ratio (last 10 steps)
        recent_rewards = sum(self.reward_history[-10:]) if self.reward_history else 0
        recent_total = min(10, len(self.reward_history))
        recent_ratio = recent_rewards / recent_total if recent_total > 0 else 0

        return {
            "rewards": self.rewards,
            "punishments": self.punishments,
            "ratio": ratio,
            "recent_ratio": recent_ratio,
            "plateau_detected": self.plateau_detected,
            "best_loss": self.best_loss,
        }
