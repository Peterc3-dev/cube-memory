#!/usr/bin/env python3
"""Real-corpus distillation smoke test on Qwen3-1.7B.

Same as `run_local_distill.py` but feeds tokenized FineWeb-Edu
samples instead of random tokens. The point is to show the loss
actually descending — random tokens have no signal, so a flat loss
on `run_local_distill.py` is meaningless.

Usage:
    source ~/rocm-gpu-test/venv/bin/activate
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    python phase1/run_local_distill_real.py [--steps 200]

Pulls ~200 short FineWeb-Edu paragraphs the first run, caches
them at /tmp/cube_memory_distill_corpus.txt for subsequent runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


CORPUS_CACHE = Path("/tmp/cube_memory_distill_corpus.txt")


def load_corpus(n_docs: int) -> list[str]:
    if CORPUS_CACHE.exists():
        text = CORPUS_CACHE.read_text()
        docs = [d for d in text.split("\n\n---\n\n") if d.strip()]
        if len(docs) >= n_docs:
            logger.info("loaded %d docs from cache %s", len(docs), CORPUS_CACHE)
            return docs[:n_docs]

    logger.info("streaming %d docs from FineWeb-Edu (first run)", n_docs)
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    docs = []
    for ex in ds:
        text = ex.get("text", "").strip()
        if 200 <= len(text) <= 4000:  # short paragraphs, fast tokenization
            docs.append(text)
            if len(docs) >= n_docs:
                break
    CORPUS_CACHE.write_text("\n\n---\n\n".join(docs))
    logger.info("cached %d docs to %s", len(docs), CORPUS_CACHE)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path.home() / "models" / "Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--swap-fraction", type=float, default=0.25)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-new", type=float, default=2e-4)
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("/tmp/cube_memory_distill_real_run.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from swap_ffn import swap_ffn_modules
    from distill import distill
    from cube_memory_layer import CubeMemoryLayer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    teacher_device = "cpu"
    logger.info("loading teacher (CPU)")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(teacher_device).eval()

    logger.info("loading student on %s", device)
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    student, swapped = swap_ffn_modules(
        student, fraction=args.swap_fraction,
        cube_kwargs={"d_codebook": 256, "d_value": 256, "m": 64,
                     "p": 3, "n_slots": 8192, "top_k": 4, "seed": 0},
    )
    logger.info("swapped student FFN at layers: %s", swapped)
    if not swapped:
        raise SystemExit("swap_ffn skipped every layer.")

    cube_param_ids = set()
    for mod in student.modules():
        if isinstance(mod, CubeMemoryLayer):
            for p in mod.parameters():
                cube_param_ids.add(id(p))
    n_frozen = n_trainable = 0
    for p in student.parameters():
        if id(p) in cube_param_ids:
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    logger.info("frozen %.1fM params, trainable %.2fM params",
                n_frozen / 1e6, n_trainable / 1e6)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    docs = load_corpus(args.n_docs)
    logger.info("tokenizing %d docs", len(docs))
    encs = tokenizer(
        docs, return_tensors="pt", padding="max_length", truncation=True,
        max_length=args.seq_len,
    )
    all_input_ids = encs["input_ids"]      # (N, L)
    all_attn      = encs["attention_mask"] # (N, L)

    def loader():
        rng = torch.Generator(device="cpu").manual_seed(0)
        n = all_input_ids.shape[0]
        for _ in range(args.steps * args.grad_accum):
            idx = torch.randint(0, n, (args.batch,), generator=rng).item()
            yield {
                "input_ids":      all_input_ids[idx:idx+args.batch],
                "attention_mask": all_attn[idx:idx+args.batch],
            }

    t0 = time.time()
    metrics = distill(
        teacher, student, loader(),
        lr_new=args.lr_new, lr_old=0.0,
        steps=args.steps, grad_accum=args.grad_accum,
        log_every=10, eval_every=0,
        device=device, teacher_device=teacher_device, dtype=dtype,
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
    logger.info("loss %.4f -> %.4f (drop=%.4f, %.1f%%) in %.1fs over %d steps",
                first, last, first - last, 100.0 * (first - last) / max(first, 1e-8),
                elapsed, args.steps)
    logger.info("OK — wrote %s", args.out)


if __name__ == "__main__":
    main()
