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