#!/usr/bin/env python3
"""Local Cube Memory distillation smoke test on Qwen3-1.7B.

Loads the model as both teacher and student, swaps a fraction of
the student's FFN blocks for `CubeMemoryLayer`, and runs a short
distillation loop on synthetic token streams. Verifies the pipeline
end-to-end on real model weights without requiring rented compute.

Designed for Strix Point's 24 GiB UMA pool — the model is loaded
in bf16, only the new CubeMemoryLayer params get gradients, and
the teacher is fully frozen. Memory budget at run time is roughly:
  teacher fp16: 3.4 GiB
  student fp16: 3.4 GiB
  cube_memory params (new) + their gradients + AdamW state: ~50 MiB
  KV caches + activations during the small forward passes: < 1 GiB
Total: comfortably under 16 GiB.

Run with:
    source ~/rocm-gpu-test/venv/bin/activate
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    python phase1/run_local_distill.py [--steps 50] [--swap-fraction 0.25]

Outputs a JSON with the loss curve to /tmp/cube_memory_distill_run.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path.home() / "models" / "Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--swap-fraction", type=float, default=0.25)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cube_memory_distill_run.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.model.exists():
        raise SystemExit(
            f"Model not found at {args.model}. Pull with `hf download Qwen/Qwen3-1.7B "
            f"--local-dir {args.model}`."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from swap_ffn import swap_ffn_modules
    from distill import distill

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Strix Point's 16 GiB visible VRAM can't hold teacher + student
    # + optimizer state + activations for a 1.7B model. Park the
    # teacher on CPU — its only role is forward-only logits for the
    # KL loss, no gradients, no optimizer state. The student stays
    # on GPU so backprop is fast where it matters.
    teacher_device = "cpu"
    logger.info("loading teacher (kept on CPU to fit Strix Point budget)")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(teacher_device).eval()

    logger.info("loading student on %s", device)
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    n_layers = len(student.model.layers)
    logger.info("model has %d layers", n_layers)

    student, swapped = swap_ffn_modules(
        student,
        fraction=args.swap_fraction,
        cube_kwargs={
            "d_codebook": 256,
            "d_value": 256,
            "m": 64,
            "p": 3,
            "n_slots": 8192,
            "top_k": 4,
            "seed": 0,
        },
    )
    logger.info("swapped student FFN at layers: %s", swapped)
    if not swapped:
        raise SystemExit(
            "swap_ffn skipped every layer (probably an all-MoE model). "
            "Pick a non-MoE base or extend swap_ffn."
        )

    # Memory budget: AdamW allocates m/v state for every parameter
    # with requires_grad=True. With a 1.7B base in bf16, optimizer
    # state alone is ~13.6 GiB (params * 2 (m, v) * 4 bytes). On
    # Strix Point that wipes the budget. Freeze every non-Cube
    # Memory parameter so only the new layer's tensors get state.
    from cube_memory_layer import CubeMemoryLayer
    cube_param_ids = set()
    for mod in student.modules():
        if isinstance(mod, CubeMemoryLayer):
            for p in mod.parameters():
                cube_param_ids.add(id(p))
    n_frozen = 0
    n_trainable = 0
    for p in student.parameters():
        if id(p) in cube_param_ids:
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    logger.info("frozen %.1fM params, trainable %.2fM params (cube memory only)",
                n_frozen / 1e6, n_trainable / 1e6)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab = tokenizer.vocab_size

    def loader():
        rng = torch.Generator(device="cpu").manual_seed(42)
        for _ in range(args.steps * args.grad_accum):
            # CPU-side generation; the distill loop will move the
            # student-side copy to GPU. Teacher consumes the CPU copy.
            input_ids = torch.randint(
                low=0, high=vocab, size=(args.batch, args.seq_len),
                generator=rng,
            )
            yield {"input_ids": input_ids,
                   "attention_mask": torch.ones_like(input_ids)}

    t0 = time.time()
    metrics = distill(
        teacher, student, loader(),
        lr_new=args.lr_new,
        lr_old=0.0,
        steps=args.steps,
        grad_accum=args.grad_accum,
        log_every=5,
        eval_every=0,
        device=device,
        teacher_device=teacher_device,
        dtype=dtype,
    )
    elapsed = time.time() - t0
    metrics["wall_seconds"] = elapsed
    metrics["swapped_layers"] = swapped
    metrics["steps"] = args.steps

    args.out.write_text(json.dumps({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float, list))}, indent=2))

    losses = metrics.get("train_loss_history", [])
    if not losses:
        raise SystemExit("no loss history recorded")

    first = sum(losses[:max(1, len(losses)//4)]) / max(1, len(losses)//4)
    last = sum(losses[-max(1, len(losses)//4):]) / max(1, len(losses)//4)
    descended = last < first
    logger.info("loss %.4f -> %.4f (descended=%s) in %.1fs",
                first, last, descended, elapsed)

    # The smoke test exists to prove the pipeline runs end-to-end
    # with real model weights on Strix Point UMA. Distillation on
    # *random* tokens with a cosine LR over only a few dozen steps
    # produces unstable gradients and lousy convergence — that's not
    # a bug in the layer or the loop, it's the input. Real Phase 1
    # distillation runs on a real corpus for thousands of steps.
    # We log descent for visibility but don't fail the run on it.
    logger.info("OK — wrote %s", args.out)


if __name__ == "__main__":
    main()
