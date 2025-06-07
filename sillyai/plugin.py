from importlib import import_module
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import inspect
import logging
from datetime import datetime
import weakref

from .ops import TensorCache, MixedPrecisionRouter

logger = logging.getLogger(__name__)

class SillyPlugin(ABC):
    """
    Abstract base class for SillyAI plugins.
    Plugins can hook into various stages of model execution and training.
    """
    def __init__(self):
        # self.model = None  # REMOVE: model property is now managed by child via weakref
        self.enabled = False
        self.name = self.__class__.__name__
        self.last_run = None

    def on_init(self, model):
        # self.model = model  # REMOVE: model property is now managed by child via weakref
        self.enabled = True
        self.last_run = datetime.now()
        logger.info(f"Plugin {self.name} initialized")

    def on_enable(self):
        """Called when the plugin is enabled."""
        self.enabled = True
        logger.info(f"Plugin {self.name} enabled")

    def on_disable(self):
        """Called when the plugin is disabled."""
        self.enabled = False
        logger.info(f"Plugin {self.name} disabled")

    def on_unload(self):
        """Called when the plugin is unloaded."""
        self.enabled = False
        logger.info(f"Plugin {self.name} unloaded")

    def before_forward(self, x):
        """Called before model.forward()"""
        return x

    def after_forward(self, x):
        """Called after model.forward()"""
        return x

    def before_backward(self, loss):
        """Called before loss.backward()"""
        return loss

    def after_backward(self):
        """Called after loss.backward()"""
        pass

    def before_optimize(self):
        """Called before optimizer.step()"""
        pass

    def after_optimize(self):
        """Called after optimizer.step()"""
        pass

    def on_epoch_start(self, epoch: int):
        """Called at the start of each training epoch"""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Called at the end of each training epoch"""
        pass

    def on_save(self) -> dict:
        """Called when model is being saved. Return dict of plugin state to save."""
        return {}

    def on_load(self, state: dict):
        """Called when model is being loaded with plugin's saved state."""
        pass

class PluginManager:
    """Manages execution of model operations through TensorCache and MixedPrecisionRouter."""
    
    def __init__(self):
        self.tensor_cache = TensorCache(max_bytes=1_000_000_000)  # 1GB cache
        self.mixed_precision = MixedPrecisionRouter()
        self.hooks = {
            'before_forward': [],
            'after_forward': [],
            'before_backward': [],
            'after_backward': [],
            'before_save': [],
            'after_load': []
        }
        
    def initialize(self, model):
        """Initialize the plugin manager with a model reference."""
        self.model = weakref.ref(model)
        self.tensor_cache.initialize()
        self.mixed_precision.initialize()
        logger.info("PluginManager initialized with TensorCache and MixedPrecisionRouter")
        
    def cleanup(self):
        """Clean up resources."""
        self.tensor_cache.cleanup()
        self.mixed_precision.cleanup()
        logger.info("PluginManager cleaned up")
        
    def call_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Call all registered hooks for a given hook name."""
        if hook_name not in self.hooks:
            return args[0] if args else None
            
        result = args[0] if args else None
        
        for hook in self.hooks[hook_name]:
            try:
                result = hook(result, *args[1:], **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {str(e)}")
                
        return result
        
    def get_states(self) -> Dict[str, Any]:
        """Get states for saving."""
        return {
            'tensor_cache': self.tensor_cache.get_state(),
            'mixed_precision': self.mixed_precision.get_state()
        }
        
    def load_states(self, states: Dict[str, Any]):
        """Load states."""
        if 'tensor_cache' in states:
            self.tensor_cache.load_state(states['tensor_cache'])
        if 'mixed_precision' in states:
            self.mixed_precision.load_state(states['mixed_precision'])
            
    def list_plugins(self) -> List[str]:
        """List available operations."""
        return ['TensorCache', 'MixedPrecisionRouter']
        
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get an operation instance by name."""
        if name == 'TensorCache':
            return self.tensor_cache
        elif name == 'MixedPrecisionRouter':
            return self.mixed_precision
        return None
        
    def has_plugin(self, name: str) -> bool:
        """Check if an operation is available."""
        return name in ['TensorCache', 'MixedPrecisionRouter']
        
    def clear(self):
        """Clear all hooks."""
        for hook_name in self.hooks:
            self.hooks[hook_name].clear()