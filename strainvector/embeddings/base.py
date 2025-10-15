from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import re



def _kmerize(sequence: str, k: int = 6) -> str:
    """Convert DNA sequence to space-separated k-mers for DNABERT v1 tokenizers.

    DNABERT v1 models (e.g., DNA_bert_6) require sequences to be preprocessed
    into overlapping k-mers before tokenization.

    Args:
        sequence: Raw DNA sequence (e.g., "ATCGATCG")
        k: k-mer size (default: 6)

    Returns:
        Space-separated k-mers (e.g., "ATCGAT TCGATC CGATCG")
    """
    return ' '.join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def _extract_kmer_size(model_name: str) -> int:
    """Extract k-mer size from DNABERT v1 model name.

    Examples:
        "zhihan1996/DNA_bert_6" -> 6
        "DNA_bert_3" -> 3
        "DNABERT-2-117M" -> 6 (default, but v2 doesn't need k-merization)

    Returns:
        k-mer size (default: 6)
    """
    match = re.search(r'DNA_?bert_?(\d+)', model_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 6  # default


class Embedder:
    """Minimal interface for sequence embedding backends."""

    def embed(self, sequences: List[str]) -> "np.ndarray":  # returns (N, D)
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


@dataclass
class DNABERT2Config:
    model_name: str = "zhihan1996/DNABERT-2-117M"
    tokenizer_name: Optional[str] = None
    pooling: str = "mean"  # cls, mean, or max
    normalize: bool = True
    precision: str = "fp32"  # fp32, fp16, bf16
    device: str = "auto"  # cpu|cuda|auto


class DNABERT2Embedder(Embedder):
    def __init__(self, cfg: DNABERT2Config):
        import platform
        import sys

        # Apply triton workaround when using DNABERT-2
        # Replace real triton with a fake module to force PyTorch attention fallback
        # This prevents flash attention compilation errors on Linux and macOS
        is_mac = platform.system() == "Darwin"
        is_linux = platform.system() == "Linux"
        is_dnabert2 = "dnabert-2" in cfg.model_name.lower()

        if is_dnabert2 and (is_mac or is_linux):
            # Create fake triton module to prevent flash attention
            # DNABERT-2 will fall back to PyTorch attention (stable, slightly slower)
            import types
            import importlib.machinery
            import os
            fake_triton = types.ModuleType("triton")
            fake_triton.__spec__ = importlib.machinery.ModuleSpec("triton", None)
            sys.modules["triton"] = fake_triton
            if is_mac:
                # Enable MPS fallback for unsupported operations on Mac
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "DNABERT2Embedder requires 'torch', 'transformers', and 'numpy' to be installed."
            ) from e

        self._np = np
        self._torch = torch

        tok_name = cfg.tokenizer_name or cfg.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if "nucleotide-transformer" in cfg.model_name.lower():
            self.model = AutoModelForMaskedLM.from_pretrained(cfg.model_name, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)


        if cfg.device == "auto":
            # Prefer Apple Silicon MPS on macOS, then CUDA, else CPU
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.model.to(self.device)
        self.model.eval()

        self.cfg = cfg
        try:
            hidden_size = int(self.model.config.hidden_size)
        except Exception:
            hidden_size = 768
        self._dim = hidden_size

        # Detect if this is DNABERT v1 (needs k-mer preprocessing)
        # DNABERT v1: DNA_bert_3, DNA_bert_6, etc.
        # DNABERT-2: DNABERT-2-117M, etc. (doesn't need k-merization)
        self._is_dnabert_v1 = bool(re.search(r'DNA_?bert_?\d+', cfg.model_name, re.IGNORECASE))
        self._kmer_size = _extract_kmer_size(cfg.model_name) if self._is_dnabert_v1 else None

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, sequences: List[str]) -> "np.ndarray":
        torch = self._torch
        np = self._np

        # Preprocess sequences for DNABERT v1 (requires k-mer tokenization)
        if self._is_dnabert_v1 and self._kmer_size:
            sequences = [_kmerize(seq, self._kmer_size) for seq in sequences]


        with torch.no_grad():
            toks = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks, output_hidden_states=True)

            # Extract hidden states from model output
            # Different models return different output formats
            last_hidden = None
            if hasattr(out, "last_hidden_state"):
                # Standard transformers models (dict-like output)
                last_hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states"):
                # Nucleotide Transformer returns all layers; take the last one
                last_hidden = out.hidden_states[-1]
            elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                # DNABERT-2 returns tuple: (hidden_states, pooled_output)
                last_hidden = out[0]

            if last_hidden is None:
                raise ValueError("No hidden state found in model output")

            if self.cfg.pooling == "cls":
                pooled = last_hidden[:, 0, :]
            elif self.cfg.pooling == "max":
                pooled = last_hidden.max(dim=1)[0]
            else:  # mean
                pooled = last_hidden.mean(dim=1)
            vecs = pooled.detach().cpu().numpy()
            if self.cfg.normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                vecs = vecs / norms

            # Explicit memory cleanup: delete tensors and clear device cache
            del toks, out, last_hidden, pooled
            if self.device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            return vecs


def get_embedder(
    model_name: str = "zhihan1996/DNABERT-2-117M",
    tokenizer_name: Optional[str] = None,
    pooling: str = "mean",
    normalize: bool = True,
    precision: str = "fp32",
    device: str = "auto",
) -> Embedder:
    """Factory for the default embedder.

    Tries DNABERT-2; if unavailable due to Triton/FlashAttention or platform
    constraints, falls back to DNABERT v1 (`zhihan1996/DNA_bert_6`). Users can
    always override via the `--model` flag.
    """
    cfg = DNABERT2Config(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        pooling=pooling,
        normalize=normalize,
        precision=precision,
        device=device,
    )
    try:
        return DNABERT2Embedder(cfg)
    except Exception as e:
        # Note: DNABERT-2 on macOS should now work via fake triton workaround
        # This fallback remains as a safety net for unexpected platform issues
        msg = str(e).lower()
        wants_dnabert2 = "dnabert-2" in model_name.lower()
        if wants_dnabert2 and ("triton" in msg or "flash" in msg):
            fallback = "zhihan1996/DNA_bert_6"
            print(
                f"[StrainVector] Unexpected platform dependency error with DNABERT-2. Falling back to {fallback}: {e}"
            )
            cfg_fallback = DNABERT2Config(
                model_name=fallback,
                tokenizer_name=tokenizer_name,
                pooling=pooling,
                normalize=normalize,
                precision=precision,
                device=device,
            )
            return DNABERT2Embedder(cfg_fallback)
        raise
