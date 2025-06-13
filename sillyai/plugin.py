import logging
import weakref
from abc import ABC
from datetime import datetime
from importlib import import_module
from typing import Any

from .ops import MixedPrecisionRouter, TensorCache

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

    def before_optimize(self):
        """Called before optimizer.step()"""

    def after_optimize(self):
        """Called after optimizer.step()"""

    def on_epoch_start(self, epoch: int):
        """Called at the start of each training epoch"""

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Called at the end of each training epoch"""

    def on_save(self) -> dict:
        """Called when model is being saved. Return dict of plugin state to save."""
        return {}

    def on_load(self, state: dict):
        """Called when model is being loaded with plugin's saved state."""


class PluginManager:
    """Manages execution of model operations through TensorCache and MixedPrecisionRouter."""

    def __init__(self):
        self.tensor_cache = TensorCache(max_bytes=1_000_000_000)  # 1GB cache
        self.mixed_precision = MixedPrecisionRouter()
        self.hooks = {
            "before_forward": [],
            "after_forward": [],
            "before_backward": [],
            "after_backward": [],
            "before_save": [],
            "after_load": [],
        }
        self.plugins = {}  # Store loaded plugins
        self.model = None

    def initialize(self, model):
        """Initialize the plugin manager with a model reference."""
        self.model = weakref.ref(model)
        self.tensor_cache.initialize()
        self.mixed_precision.initialize()
        logger.info(
            "PluginManager initialized with TensorCache and MixedPrecisionRouter",
        )

    def load_plugin(self, plugin_name: str) -> bool:
        """Load and initialize a plugin by name.

        Args:
            plugin_name: Name of the plugin to load

        Returns:
            bool: True if plugin was loaded successfully, False otherwise
        """
        try:
            # Check if plugin is already loaded
            if plugin_name in self.plugins:
                logger.info(f"Plugin {plugin_name} is already loaded")
                return True

            # Import and instantiate plugin
            module_name = f"sillyai.plugins.{plugin_name.lower()}"
            try:
                module = import_module(module_name)
                plugin_class = getattr(module, f"{plugin_name}Plugin")
                plugin = plugin_class()
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e!s}")
                return False

            # Initialize plugin with model reference
            if self.model is not None:
                plugin.on_init(self.model())

            # Store plugin instance
            self.plugins[plugin_name] = plugin

            # Register plugin hooks
            for hook_name in self.hooks.keys():
                hook_method = getattr(plugin, hook_name, None)
                if hook_method is not None:
                    self.hooks[hook_name].append(hook_method)

            logger.info(f"Successfully loaded plugin {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e!s}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            bool: True if plugin was unloaded successfully, False otherwise
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} is not loaded")
            return False

        try:
            plugin = self.plugins[plugin_name]

            # Remove plugin hooks
            for hook_name in self.hooks.keys():
                hook_method = getattr(plugin, hook_name, None)
                if hook_method in self.hooks[hook_name]:
                    self.hooks[hook_name].remove(hook_method)

            # Call plugin cleanup
            plugin.on_unload()

            # Remove plugin instance
            del self.plugins[plugin_name]

            logger.info(f"Successfully unloaded plugin {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e!s}")
            return False

    def cleanup(self):
        """Clean up resources."""
        # Unload all plugins
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)

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
                logger.error(f"Error in hook {hook_name}: {e!s}")

        return result

    def get_states(self) -> dict[str, Any]:
        """Get states for saving."""
        return {
            "tensor_cache": self.tensor_cache.get_state(),
            "mixed_precision": self.mixed_precision.get_state(),
        }

    def load_states(self, states: dict[str, Any]):
        """Load states."""
        if "tensor_cache" in states:
            self.tensor_cache.load_state(states["tensor_cache"])
        if "mixed_precision" in states:
            self.mixed_precision.load_state(states["mixed_precision"])

    def list_plugins(self) -> list[str]:
        """List available operations."""
        return ["TensorCache", "MixedPrecisionRouter"]

    def get_plugin(self, name: str) -> Any | None:
        """Get an operation instance by name."""
        if name == "TensorCache":
            return self.tensor_cache
        elif name == "MixedPrecisionRouter":
            return self.mixed_precision
        return None

    def has_plugin(self, name: str) -> bool:
        """Check if an operation is available."""
        return name in ["TensorCache", "MixedPrecisionRouter"]

    def clear(self):
        """Clear all hooks."""
        for hook_name in self.hooks:
            self.hooks[hook_name].clear()
