from typing import Optional, List, Any

class ModelConfig:
    def __init__(
        self,
        output_dim=1,
        concept_dim=16,
        input_dim=1,
        d_model=32,
        num_layers=2,
        nhead=2,
        dim_ff=64,
        hidden_dim=128,
        kronecker_rank=16,  # Set a reasonable default
        optim_args=None,
        snapshot_dir="./snapshots",
        **kwargs
    ):
        # Model architecture
        self.output_dim = output_dim
        self.concept_dim = concept_dim
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_ff = dim_ff
        self.hidden_dim = hidden_dim
        self.factorized_linear = kwargs.get('factorized_linear', True)
        self.kronecker_rank = kronecker_rank  # Now always available
        
        # Training settings
        self.optim_args = optim_args or {
            'use_amp': True,
            'grad_clip': 1.0,
            'lr': 1e-4,
            'warmup_steps': 1000,
            'keep_best_only': True
        }
        
        # Additional params
        self.concept_loss_weight = kwargs.get('concept_loss_weight', 0.1)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        self.plugin_dir = kwargs.get('plugin_dir', './plugins')
        self.snapshot_dir = snapshot_dir
        
        # InfiniToeplitz settings
        self.infini_local_window = kwargs.get('infini_local_window', 512)
        self.infini_mem_size = kwargs.get('infini_mem_size', 1024)
        self.infini_compress_ratio = kwargs.get('infini_compress_ratio', 4)