"""Embedding backends for StrainVector (optional deps).

This package provides a small factory to obtain an embedder. By default,
it attempts to instantiate a DNABERT-2-based embedder via HuggingFace
Transformers and PyTorch if they are available.
"""

from .base import Embedder, get_embedder  # noqa: F401

