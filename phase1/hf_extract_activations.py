#!/usr/bin/env python3
"""Extract per-layer FFN (x_in, x_out) activation pairs from a HuggingFace
causal-LM and write them in the CACT chunk format consumed by
per_layer_trainer.load_layer_pairs / reviewer_exp1_svd_spectrum.

This is the thinkhub-CPU substitute for the GPD's llama-dump-activations tool:
it lets Case Study 1 be reproduced on a second model (e.g. Qwen3-4B) without
GGUF or the patched llama.cpp. It hooks each chosen decoder layer's `mlp`
submodule, capturing the MLP input (FFN x_in) and MLP output (FFN x_out).

Byte format (matches per_layer_trainer._read_chunk):
  header = struct '<4sIII' = (b"CACT", version=1, n_tokens, n_embd)
  payload = n_tokens * n_embd bfloat16 values (uint16 wire), row-major.

Usage:
  python phase1/hf_extract_activations.py \
    --model Qwen/Qwen3-4B --layers 3 18 32 \
    --target-tokens 12000 --seq-len 512 \
    --out-dir ~/cube-memory-cache/activations-qwen3-4b
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CACT_MAGIC = b"CACT"
CACT_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIII")


def write_chunk(path: Path, tensor: torch.Tensor) -> None:
    """tensor: (n_tokens, n_embd) on CPU. Stored as bf16 raw bytes."""
    n_tokens, n_embd = tensor.shape
    bf16 = tensor.detach().to(torch.bfloat16).contiguous()
    raw_u16 = bf16.view(torch.uint16)  # reinterpret 16 bits, no value change
    with path.open("wb") as f:
        f.write(HEADER_STRUCT.pack(CACT_MAGIC, CACT_VERSION, n_tokens, n_embd))
        f.write(raw_u16.numpy().tobytes())


def find_decoder_layers(model):
    """Standard HF layout: model.model.layers (ModuleList)."""
    m = getattr(model, "model", model)
    layers = getattr(m, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    return layers


def build_corpus(target_tokens: int, tokenizer, corpus_file: str | None = None) -> list[str]:
    """Use an explicit corpus file (one doc per line) if given; else stream
    FineWeb-Edu; else fall back to local .md. The explicit file lets a second
    model be extracted on the SAME corpus the 27B used (fineweb_100k.txt)."""
    if corpus_file:
        p = Path(corpus_file).expanduser()
        docs = [ln for ln in p.read_text(errors="ignore").splitlines() if ln.strip()]
        logger.info("explicit corpus %s: %d docs", p, len(docs))
        return docs
    docs: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        tok_count = 0
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            docs.append(text)
            tok_count += len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if tok_count >= target_tokens * 1.2:
                break
        logger.info("FineWeb-Edu: %d docs (~%d tokens)", len(docs), tok_count)
        return docs
    except Exception as e:  # network/datasets unavailable
        logger.warning("FineWeb streaming failed (%s); falling back to local corpus", e)
        roots = [Path.home() / "CIN", Path.home() / "projects" / "cube-memory"]
        for root in roots:
            if root.exists():
                for p in root.rglob("*.md"):
                    try:
                        docs.append(p.read_text(errors="ignore"))
                    except Exception:
                        pass
        logger.info("local fallback: %d .md docs", len(docs))
        return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--target-tokens", type=int, default=12000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corpus-file", default=None,
                    help="explicit corpus, one doc per line (e.g. fineweb_100k.txt)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading tokenizer + model %s (CPU, bf16)...", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    logger.info("model loaded: hidden=%d, layers=%d", d_model, n_layers)
    for L in args.layers:
        if L >= n_layers:
            logger.error("layer %d >= num_hidden_layers %d", L, n_layers)
            return 2

    layers = find_decoder_layers(model)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            x_in = inputs[0]
            x_out = output[0] if isinstance(output, tuple) else output
            captured[idx] = (x_in.detach().to(torch.float32),
                             x_out.detach().to(torch.float32))
        return hook

    handles = [layers[L].mlp.register_forward_hook(make_hook(L)) for L in args.layers]

    # prep output dirs
    for L in args.layers:
        for side in ("in", "out"):
            (out_dir / f"layer_{L}_{side}").mkdir(parents=True, exist_ok=True)

    docs = build_corpus(args.target_tokens, tok, args.corpus_file)
    if not docs:
        logger.error("empty corpus")
        return 3

    tokens_done = 0
    chunk_idx = 0
    t0 = time.time()
    with torch.no_grad():
        for doc in docs:
            ids = tok(doc, add_special_tokens=False, truncation=True,
                      max_length=args.seq_len, return_tensors="pt")["input_ids"]
            if ids.shape[1] < 8:
                continue
            captured.clear()
            model(ids)  # batch=1, no padding -> every position is a real token
            for L in args.layers:
                x_in, x_out = captured[L]
                write_chunk(out_dir / f"layer_{L}_in" / f"chunk_{chunk_idx:05d}.bin",
                            x_in.reshape(-1, d_model))
                write_chunk(out_dir / f"layer_{L}_out" / f"chunk_{chunk_idx:05d}.bin",
                            x_out.reshape(-1, d_model))
            chunk_idx += 1
            tokens_done += ids.shape[1]
            if chunk_idx % 5 == 0 or tokens_done >= args.target_tokens:
                rate = tokens_done / (time.time() - t0)
                logger.info("tokens=%d/%d  seqs=%d  %.1f tok/s",
                            tokens_done, args.target_tokens, chunk_idx, rate)
            if tokens_done >= args.target_tokens:
                break

    for h in handles:
        h.remove()
    logger.info("DONE: %d tokens, %d chunks/layer, d_model=%d, layers=%s -> %s",
                tokens_done, chunk_idx, d_model, args.layers, out_dir)
    # leave a small manifest
    (out_dir / "MANIFEST.txt").write_text(
        f"model={args.model}\nd_model={d_model}\nn_layers={n_layers}\n"
        f"layers={args.layers}\ntokens={tokens_done}\nseq_len={args.seq_len}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
