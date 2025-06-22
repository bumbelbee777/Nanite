from .ops import MultivectorOps
from .utils import DebugLogger

# Create a shared instance of MultivectorOps
_shared_ops = MultivectorOps()
_shared_ops.compile()
_shared_debug = DebugLogger()


def get_shared_ops() -> MultivectorOps:
    """Get the shared instance of MultivectorOps."""
    return _shared_ops
