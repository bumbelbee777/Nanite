import os
import importlib.util
from abc import ABC, abstractmethod

class SillyPlugin(ABC):
    """
    Base class for all SillyAI plugins.
    Plugins can hook into init, forward, train_step, etc.
    """

    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""
        ...

    def on_init(self, model):
        """Called at end of SillyAI.__init__."""
        pass

    def before_forward(self, model, x):
        """Called at the very start of model.forward(x).
        Return potentially modified x."""
        return x

    def after_forward(self, model, x, output):
        """Called after model.forward, before returning.
        Return potentially modified output."""
        return output

    def on_train_step(self, model, batch, loss):
        """Called at the end of each train_step."""
        pass

    def on_validation_step(self, model, batch, loss):
        """Called at the end of each validation_step."""
        pass

    def on_concept_update(self, model, concept_name, concept):
        """Called when a concept is updated."""
        pass

class PluginManager:
    def __init__(self, plugin_dir: str):
        self.plugin_dir = plugin_dir
        self.plugins: list[SillyPlugin] = []

    def discover(self):
        """Dynamically import all .py files in plugin_dir that define SillyPlugin subclasses."""
        if not os.path.isdir(self.plugin_dir):
            return

        for fname in os.listdir(self.plugin_dir):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            path = os.path.join(self.plugin_dir, fname)
            spec = importlib.util.spec_from_file_location(fname[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # find plugin classes
            for obj in mod.__dict__.values():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, SillyPlugin)
                    and obj is not SillyPlugin
                ):
                    inst = obj()
                    self.plugins.append(inst)

    def apply_on_init(self, model):
        for p in self.plugins:
            p.on_init(model)

    def apply_before_forward(self, model, x):
        for p in self.plugins:
            x = p.before_forward(model, x)
        return x

    def apply_after_forward(self, model, x, output):
        for p in self.plugins:
            output = p.after_forward(model, x, output)
        return output

    def apply_on_train_step(self, model, batch, loss):
        for p in self.plugins:
            p.on_train_step(model, batch, loss)

    def apply_on_validation_step(self, model, batch, loss):
        for p in self.plugins:
            p.on_validation_step(model, batch, loss)
