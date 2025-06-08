import torch

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
        self.rewards = 0
        self.punishments = 0
        
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
            self.rewards += 1
        else:
            # Punishment: Decrease learning rate
            self.current_lr = max(self.current_lr * self.punishment_factor, self.min_lr)
            self.no_improvement_count += 1
            self.punishments += 1
            
        # Update optimizer
        self._update_optimizer(optimizer)
        
    def _update_optimizer(self, optimizer):
        """Update optimizer learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr
            
    def get_reward_stats(self):
        """Get statistics about rewards and punishments."""
        total = self.rewards + self.punishments
        ratio = self.rewards / total if total > 0 else 0
        
        return {
            "rewards": self.rewards,
            "punishments": self.punishments,
            "ratio": ratio
        } 