"""distill — knowledge distillation for the Cube Memory student.

Teacher: original (frozen) HF causal LM.
Student: same model with a subset of FFN blocks replaced by
CubeMemoryLayer instances (see `swap_ffn.swap_ffn_modules`).

Loss: temperature-scaled KL between teacher and student logits,
averaged over unmasked positions. Optimizer: AdamW with two parameter
groups so the new CubeMemoryLayer params train at a higher LR than any
unfrozen pre-existing params (e.g. layer norms around swapped blocks).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube_memory_layer import CubeMemoryLayer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Param discovery / freezing
# ----------------------------------------------------------------------

def _cube_layer_param_ids(student: nn.Module) -> set[int]:
    """Set of `id(p)` for every parameter that lives inside a CubeMemoryLayer."""
    ids: set[int] = set()
    for mod in student.modules():
        if isinstance(mod, CubeMemoryLayer):
            for p in mod.parameters():
                ids.add(id(p))
    return ids


def _cube_layer_indices(student: nn.Module) -> list[int]:
    """Indices in `model.model.layers` whose `.mlp` is a CubeMemoryLayer.

    Returns [] if the standard layer-list attribute is missing.
    """
    inner = getattr(student, "model", None)
    if inner is None:
        return []
    layers = getattr(inner, "layers", None)
    if layers is None:
        return []
    out = []
    for i, layer in enumerate(layers):
        if isinstance(getattr(layer, "mlp", None), CubeMemoryLayer):
            out.append(i)
    return out


def _unfreeze_norms_around(student: nn.Module, indices: list[int]) -> list[nn.Parameter]:
    """Unfreeze layer-norm params on swapped blocks. Returns the list of
    parameters we just enabled gradients on."""
    inner = getattr(student, "model", None)
    if inner is None:
        return []
    layers = getattr(inner, "layers", None)
    if layers is None:
        return []
    enabled: list[nn.Parameter] = []
    for i in indices:
        layer = layers[i]
        for name, mod in layer.named_modules():
            # Heuristic: anything called *_layernorm or *_norm or matching
            # nn.LayerNorm / RMSNorm-likes.
            cls = type(mod).__name__.lower()
            if "norm" in cls:
                for p in mod.parameters(recurse=False):
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        enabled.append(p)
    return enabled


def _build_param_groups(
    student: nn.Module,
    lr_new: float,
    lr_old: float,
    unfreeze_norms: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Two param groups: CubeMemoryLayer params (lr_new) and any other
    still-trainable params (lr_old). Returns (param_groups, swapped_indices).
    """
    swapped = _cube_layer_indices(student)

    if unfreeze_norms and swapped:
        added = _unfreeze_norms_around(student, swapped)
        if added:
            logger.info("unfroze %d layer-norm params on swapped blocks", len(added))

    cube_ids = _cube_layer_param_ids(student)
    new_params: list[nn.Parameter] = []
    old_params: list[nn.Parameter] = []
    for p in student.parameters():
        if not p.requires_grad:
            continue
        if id(p) in cube_ids:
            new_params.append(p)
        else:
            old_params.append(p)

    groups: list[dict[str, Any]] = []
    if new_params:
        groups.append({"params": new_params, "lr": lr_new, "name": "new"})
    if old_params:
        groups.append({"params": old_params, "lr": lr_old, "name": "old"})
    if not groups:
        raise RuntimeError(
            "No trainable parameters found on student. "
            "Did you call swap_ffn_modules() before distill()?"
        )
    return groups, swapped


# ----------------------------------------------------------------------
# LR schedule
# ----------------------------------------------------------------------

def _cosine_factor(step: int, total: int) -> float:
    """Cosine decay from 1.0 at step=0 to 0.0 at step=total."""
    if total <= 1:
        return 1.0
    progress = min(max(step / max(total - 1, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def _kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """KL(student || teacher) under temperature T, averaged over unmasked
    positions. Multiplied by T**2 (Hinton 2015) so the gradient magnitude
    is invariant to T.

    Shapes: logits are (B, T, V), mask is (B, T) (1 = keep, 0 = pad).
    """
    T = temperature
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    t_p = t_logp.exp()
    # Per-token KL: sum over vocab.
    per_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)  # (B, T)
    if attention_mask is not None:
        mask = attention_mask.to(per_tok.dtype)
        denom = mask.sum().clamp(min=1.0)
        loss = (per_tok * mask).sum() / denom
    else:
        loss = per_tok.mean()
    return loss * (T * T)


# ----------------------------------------------------------------------
# OOM helpers
# ----------------------------------------------------------------------

def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "hip out of memory" in msg


def _halve_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    """Return a dict with the leading dim halved, or None if it would be empty."""
    bsz = next(iter(batch.values())).shape[0]
    if bsz <= 1:
        return None
    new_bsz = bsz // 2
    return {k: v[:new_bsz] for k, v in batch.items()}


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------

def _grad_norm(params: Iterator[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _data_iter(dataloader) -> Iterator[dict[str, torch.Tensor]]:
    """Cycle the dataloader indefinitely."""
    while True:
        for batch in dataloader:
            yield batch


def distill(
    teacher: nn.Module,
    student: nn.Module,
    dataloader,
    *,
    lr_new: float = 1e-4,
    lr_old: float = 1e-5,
    steps: int = 1000,
    grad_accum: int = 4,
    kl_temperature: float = 2.0,
    log_every: int = 10,
    eval_every: int = 100,
    eval_fn: Callable[[nn.Module], dict[str, float]] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    unfreeze_norms: bool = False,
    teacher_device: str | None = None,
) -> dict[str, Any]:
    """Distill the teacher's logits into the student.

    See module docstring for design notes. Returns a metrics dict
    containing the final/last train loss, last grad norm, last LRs,
    and (if `eval_fn` was provided) the last eval result.

    Set `teacher_device` to a different string (e.g., "cpu") to keep
    the teacher off the student's device — useful on memory-tight
    setups like Strix Point UMA.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")

    dev = torch.device(device)
    t_dev = torch.device(teacher_device) if teacher_device is not None else dev

    # Freeze teacher fully.
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(t_dev)

    # Student must be in training mode so the unit-modulus / STE path
    # in CubeMemoryLayer.forward executes its training-time semantics.
    student.train()
    student.to(dev)

    # Cache teacher params + their tensor identities for the no-mutation check.
    teacher_param_ids = {id(p): p.detach().clone() for p in teacher.parameters()}

    param_groups, swapped_indices = _build_param_groups(
        student, lr_new=lr_new, lr_old=lr_old, unfreeze_norms=unfreeze_norms
    )
    base_lrs = [g["lr"] for g in param_groups]
    optimizer = torch.optim.AdamW(param_groups)

    autocast_enabled = dev.type in ("cuda", "hip") and dtype in (torch.bfloat16, torch.float16)

    metrics: dict[str, Any] = {
        "train_loss_history": [],
        "swapped_indices": list(swapped_indices),
        "last_eval": None,
    }

    iterator = _data_iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    micro_in_step = 0
    step = 0

    while step < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        batch = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        attempt_batch: dict[str, torch.Tensor] | None = batch

        # Inner loop handles OOM by halving batch size in place.
        while attempt_batch is not None:
            try:
                input_ids = attempt_batch["input_ids"]
                attn_mask = attempt_batch.get("attention_mask")

                # Move inputs to each model's device. With teacher_device
                # set (e.g. "cpu"), the teacher runs on its own hardware
                # and we ferry just the logits over to the student device
                # for the loss.
                t_input_ids = input_ids.to(t_dev)
                t_attn_mask = attn_mask.to(t_dev) if attn_mask is not None else None
                s_input_ids = input_ids.to(dev)
                s_attn_mask = attn_mask.to(dev) if attn_mask is not None else None

                with torch.no_grad():
                    if autocast_enabled and t_dev.type == dev.type:
                        with torch.autocast(device_type=dev.type, dtype=dtype):
                            t_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)
                    else:
                        t_out = teacher(input_ids=t_input_ids, attention_mask=t_attn_mask)
                    t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
                    if t_logits.device != dev:
                        t_logits = t_logits.to(dev)

                if autocast_enabled:
                    with torch.autocast(device_type=dev.type, dtype=dtype):
                        s_out = student(input_ids=s_input_ids, attention_mask=s_attn_mask)
                        s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
                        loss = _kl_loss(s_logits, t_logits, s_attn_mask, kl_temperature)
                else:
                    s_out = student(input_ids=s_input_ids, attention_mask=s_attn_mask)
                    s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
                    loss = _kl_loss(s_logits, t_logits, s_attn_mask, kl_temperature)

                (loss / grad_accum).backward()
                accum_loss += loss.detach().float().item()
                micro_in_step += 1
                break  # success
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                # Clear any half-built state and try a smaller micro-batch.
                optimizer.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                halved = _halve_batch(attempt_batch)
                if halved is None:
                    raise RuntimeError(
                        "CUDA OOM at minimum batch size 1; cannot recover."
                    ) from e
                logger.warning(
                    "OOM caught; halving micro-batch to %d and retrying",
                    next(iter(halved.values())).shape[0],
                )
                # Note: this resets accumulation for the current step, so
                # train_loss for the step that hit OOM only reflects the
                # post-recovery micro-batches. Acceptable on rare OOMs;
                # if it matters, track an oom_recovered flag in metrics.
                attempt_batch = halved
                accum_loss = 0.0
                micro_in_step = 0
                continue

        if micro_in_step < grad_accum:
            continue  # accumulate more

        # Optimizer step.
        gn = _grad_norm(p for g in optimizer.param_groups for p in g["params"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Cosine decay. Indexed from 0 so step 0 sees factor=1.0 (max LR).
        factor = _cosine_factor(step, steps)
        for grp, base in zip(optimizer.param_groups, base_lrs):
            grp["lr"] = base * factor

        avg_loss = accum_loss / max(micro_in_step, 1)
        metrics["train_loss_history"].append(avg_loss)
        accum_loss = 0.0
        micro_in_step = 0

        if (step % log_every == 0) or (step == steps - 1):
            lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}
            logger.info(
                "step=%d loss=%.4f grad_norm=%.4f lrs=%s",
                step, avg_loss, gn, lrs,
            )

        if eval_fn is not None and eval_every > 0 and ((step + 1) % eval_every == 0):
            student.eval()
            with torch.no_grad():
                metrics["last_eval"] = eval_fn(student)
            student.train()
            logger.info("step=%d eval=%s", step, metrics["last_eval"])

        step += 1

    # Verify teacher didn't drift.
    drift = 0.0
    for p in teacher.parameters():
        ref = teacher_param_ids.get(id(p))
        if ref is None:
            continue
        drift = max(drift, (p.detach() - ref.to(p.device)).abs().max().item())
    metrics["teacher_max_drift"] = drift
    if drift > 0.0:
        logger.warning("teacher drifted by %.3e — should be 0", drift)

    metrics["final_train_loss"] = (
        metrics["train_loss_history"][-1] if metrics["train_loss_history"] else float("nan")
    )
    return metrics


__all__ = ["distill"]
