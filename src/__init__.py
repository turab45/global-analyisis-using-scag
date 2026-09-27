"""Global semantic coactivation analysis of fixed neural networks."""

from .alignment import compute_alignment, compute_semantic_similarity
from .coactivation import compute_correlation

# Keep the spelling from the original API request available as an alias.
compute_correaltion = compute_correlation

__all__ = ["compute_correlation", "compute_correaltion", "compute_semantic_similarity", "compute_alignment"]
