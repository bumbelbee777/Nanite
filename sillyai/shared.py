from .ops import MultivectorOps

# Create a shared instance of MultivectorOps
_shared_ops = MultivectorOps().compile()


def get_shared_ops() -> MultivectorOps:
    """Get the shared instance of MultivectorOps."""
    return _shared_ops
