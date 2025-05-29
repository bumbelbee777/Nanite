from importlib import import_module
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import inspect
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SillyPlugin(ABC):
    """
    Abstract base class for SillyAI plugins.
    Plugins can hook into various stages of model execution and training.
    """
    def __init__(self):
        self.model = None
        self.enabled = False
        self.name = self.__class__.__name__
        self.last_run = None

    def on_init(self, model):
        """Called when the plugin is initialized with the model."""
        self.model = model
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
        self.model = None
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
    """Manages SillyAI plugins lifecycle and execution"""
    def __init__(self, model):
        self.model = model
        self.plugins: Dict[str, SillyPlugin] = {}
        self.active_plugins: List[str] = []
        self._hook_cache = {}
        logger.info("Plugin manager initialized")

    def discover_plugins(self, plugin_dir: str = "plugins"):
        """Discover plugins in the specified directory"""
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return

        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                # Import module
                module_name = f"plugins.{py_file.stem}"
                module = import_module(module_name)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, SillyPlugin) and 
                        obj != SillyPlugin):
                        
                        plugin = obj()
                        self.register_plugin(plugin)
                        
            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}")

    def register_plugin(self, plugin: SillyPlugin):
        """Register a plugin instance"""
        name = plugin.__class__.__name__
        if name in self.plugins:
            logger.warning(f"Plugin {name} already registered")
            return False
            
        self.plugins[name] = plugin
        logger.info(f"Registered plugin: {name}")
        return True

    def enable_plugin(self, name: str):
        """Enable a registered plugin"""
        if name not in self.plugins:
            logger.error(f"Plugin {name} not found")
            return False
            
        plugin = self.plugins[name]
        if name not in self.active_plugins:
            plugin.on_init(self.model)
            self.active_plugins.append(name)
            plugin.on_enable()
        return True

    def disable_plugin(self, name: str):
        """Disable an active plugin"""
        if name in self.active_plugins:
            plugin = self.plugins[name]
            plugin.on_disable()
            self.active_plugins.remove(name)
            return True
        return False

    def unload_plugin(self, name: str):
        """Completely unload a plugin"""
        if name in self.active_plugins:
            self.disable_plugin(name)
        if name in self.plugins:
            plugin = self.plugins[name]
            plugin.on_unload()
            del self.plugins[name]
            return True
        return False

    def get_plugin(self, name: str) -> Optional[SillyPlugin]:
        """Get a plugin instance by name"""
        return self.plugins.get(name)

    def call_hook(self, hook_name: str, *args, **kwargs):
        """Call a specific hook on all active plugins"""
        results = []
        for name in self.active_plugins:
            plugin = self.plugins[name]
            hook = getattr(plugin, hook_name, None)
            if hook and callable(hook):
                try:
                    result = hook(*args, **kwargs)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in plugin {name} hook {hook_name}: {e}")
                    
        return results[0] if results else args[0] if args else None

    def save_states(self) -> Dict[str, dict]:
        """Get save states from all active plugins"""
        states = {}
        for name in self.active_plugins:
            plugin = self.plugins[name]
            try:
                states[name] = plugin.on_save()
            except Exception as e:
                logger.error(f"Error saving state for plugin {name}: {e}")
        return states

    def load_states(self, states: Dict[str, dict]):
        """Load saved states into plugins"""
        for name, state in states.items():
            plugin = self.plugins.get(name)
            if plugin:
                try:
                    plugin.on_load(state)
                except Exception as e:
                    logger.error(f"Error loading state for plugin {name}: {e}")