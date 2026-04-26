"""swap_ffn — replace dense FFN modules in HF transformers with CubeMemoryLayer.

Targets the Qwen3-class architecture (the standard HF layout where the
top-level module is `model.model` and its `.layers` is an `nn.ModuleList`
of decoder blocks, each with a `.mlp` attribute).

MoE blocks are detected by class name and skipped — the CubeMemoryLayer
is a dense-FFN drop-in, not an expert-routing replacement. The caller
gets back the actually-replaced indices so they know what changed.
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

from cube_memory_layer import CubeMemoryLayer

logger = logging.getLogger(__name__)


# Class-name fragments that flag a sparse-MoE block we must NOT swap.
# Matched via substring so we cover Qwen3MoeSparseMoeBlock,
# Qwen3NextSparseMoeBlock, Qwen35MoeSparseMoeBlock, etc.
_MOE_NAME_HINTS = ("MoeSparse", "SparseMoe", "MoeBlock", "MoESparse")


def _is_moe_module(mod: nn.Module) -> bool:
    """True if `mod` looks like a sparse-MoE FFN block we should skip."""
    name = type(mod).__name__
    return any(hint in name for hint in _MOE_NAME_HINTS)


def _evenly_spaced(n_layers: int, fraction: float) -> list[int]:
    """Pick `round(n_layers * fraction)` indices spread across [0, n_layers).

    Spacing matches the PLAN.md example: for n_layers=48 and fraction=0.25
    return [3, 7, 11, ..., 47] — i.e. last-of-each-block.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = max(1, round(n_layers * fraction))
    if k > n_layers:
        k = n_layers
    stride = n_layers // k
    # End-of-block indexing: i*stride + (stride - 1).
    indices = [i * stride + (stride - 1) for i in range(k)]
    # Clip to range; should already be inside but guard against rounding.
    indices = [min(i, n_layers - 1) for i in indices]
    return indices


def _resolve_layer_list(model: nn.Module) -> nn.ModuleList:
    """Find the standard HF Qwen `model.model.layers` ModuleList."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise AttributeError(
            "swap_ffn_modules expects a HF-style model with a `.model` "
            "attribute (e.g. AutoModelForCausalLM). Got "
            f"{type(model).__name__!r} with no `.model`."
        )
    layers = getattr(inner, "layers", None)
    if layers is None or not isinstance(layers, (nn.ModuleList, list)):
        raise AttributeError(
            "swap_ffn_modules expects `model.model.layers` to be an "
            "nn.ModuleList of decoder blocks. Found "
            f"{type(layers).__name__!r}."
        )
    return layers


def _hidden_size(model: nn.Module, layer: nn.Module) -> int:
    """Best-effort recovery of the hidden dim for a CubeMemoryLayer instance.

    Tries (in order): `model.config.hidden_size`, then the input feature
    of the layer's existing `.mlp` if it exposes a recognisable input
    Linear (`gate_proj`/`up_proj`/`fc1`/`in_features`).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for attr in ("gate_proj", "up_proj", "fc1"):
            sub = getattr(mlp, attr, None)
            if isinstance(sub, nn.Linear):
                return int(sub.in_features)
        if hasattr(mlp, "in_features"):
            return int(mlp.in_features)
    raise RuntimeError(
        "Could not infer hidden size for CubeMemoryLayer. Pass "
        "`cube_kwargs={'d_in': <int>}` explicitly."
    )


def _module_dtype_device(mod: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Pull dtype/device from the first parameter of `mod`.

    Falls back to (float32, cpu) if `mod` has no parameters.
    """
    for p in mod.parameters():
        return p.dtype, p.device
    for b in mod.buffers():
        return b.dtype, b.device
    return torch.float32, torch.device("cpu")


def swap_ffn_modules(
    model: nn.Module,
    layer_indices: list[int] | None = None,
    cube_kwargs: dict | None = None,
    fraction: float = 0.25,
) -> tuple[nn.Module, list[int]]:
    """Replace `.mlp` on selected decoder layers with `CubeMemoryLayer`.

    Args:
        model: HF-style transformer (must expose `model.model.layers`).
        layer_indices: explicit indices to swap. If None, evenly spaced
            picks covering `fraction` of the layer count.
        cube_kwargs: forwarded to CubeMemoryLayer. `d_in` is auto-filled
            from the model config if not provided.
        fraction: used only when `layer_indices is None`.

    Returns:
        (model, replaced_indices). `replaced_indices` is the actual
        list of layers that were swapped (MoE blocks are skipped, so
        this can be shorter than the requested set).
    """
    cube_kwargs = dict(cube_kwargs) if cube_kwargs else {}
    layers = _resolve_layer_list(model)
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError("model.model.layers is empty.")

    if layer_indices is None:
        candidates: list[int] = _evenly_spaced(n_layers, fraction)
    else:
        candidates = list(layer_indices)
        for idx in candidates:
            if not 0 <= idx < n_layers:
                raise IndexError(
                    f"layer index {idx} out of range for {n_layers} layers"
                )

    replaced: list[int] = []
    for idx in candidates:
        layer = layers[idx]
        old_mlp = getattr(layer, "mlp", None)
        if old_mlp is None:
            logger.warning(
                "layer %d has no `.mlp` attribute; skipping", idx
            )
            continue
        if _is_moe_module(old_mlp):
            logger.warning(
                "layer %d mlp is %s (MoE block); skipping",
                idx,
                type(old_mlp).__name__,
            )
            continue

        kwargs = dict(cube_kwargs)
        if "d_in" not in kwargs:
            kwargs["d_in"] = _hidden_size(model, layer)

        dtype, device = _module_dtype_device(old_mlp)
        cube = CubeMemoryLayer(**kwargs)
        # Cast real params/buffers to match the host dtype; leave complex
        # codebook buffers in their native complex dtype (a naive .to(dtype=)
        # would cast complex64 -> real and silently discard the imaginary
        # part, breaking the FHRR algebra).
        for name, param in cube.named_parameters(recurse=True):
            param.data = param.data.to(dtype=dtype, device=device)
        for name, buf in cube.named_buffers(recurse=True):
            if buf.is_complex():
                buf.data = buf.data.to(device=device)
            else:
                buf.data = buf.data.to(dtype=dtype, device=device)

        layer.mlp = cube
        replaced.append(idx)
        logger.info(
            "swapped layer %d: %s -> CubeMemoryLayer(d_in=%d)",
            idx,
            type(old_mlp).__name__,
            kwargs["d_in"],
        )

    return model, replaced


__all__ = ["swap_ffn_modules"]
