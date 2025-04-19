import os
import importlib.util
from plugin import SillyPlugin

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
