#!/usr/bin/env python3
"""Sample FineWeb-Edu text into a flat corpus file for llama-dump-activations.

Usage:
    ~/rocm-gpu-test/venv/bin/python phase1/fineweb_sample.py \
        --target-tokens 100000 \
        --out ~/cube-memory-cache/corpus/fineweb_100k.txt

Streams from HuggingFaceFW/fineweb-edu (sample-10BT split) and writes one
document per line until the Qwen3 tokenizer count exceeds --target-tokens.
The output file is consumed by `llama-dump-activations -f <file>`; the line
delimiter is irrelevant to that tool (it tokenizes the whole file as text).

Falls back to a local synthetic corpus (concatenated .md files from
~/CIN/ and ~/projects/cube-memory/) if `datasets` cannot reach HF.

This is Phase A *prep*. We do not actually run the FineWeb download here
unless invoked directly; the script simply must be ready.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_OUT = Path.home() / "cube-memory-cache" / "corpus" / "fineweb_100k.txt"
DEFAULT_TOKENIZER = Path.home() / "models" / "Qwen3-1.7B"


def _gather_local_fallback() -> list[str]:
    """Concatenate any .md content under ~/CIN/ and ~/projects/cube-memory/
    for use when HF streaming is unavailable. Strictly a bootstrap stub —
    not representative training data."""
    roots = [Path.home() / "CIN", Path.home() / "projects" / "cube-memory"]
    docs: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for md in root.rglob("*.md"):
            try:
                txt = md.read_text(errors="ignore").strip()
            except OSError:
                continue
            if len(txt) >= 200:
                docs.append(txt)
    return docs


def _stream_fineweb(min_chars: int = 200, max_chars: int = 4000):
    """Yield text strings from HuggingFaceFW/fineweb-edu (sample-10BT)."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if min_chars <= len(text) <= max_chars:
            yield text


def sample(
    target_tokens: int,
    out_path: Path,
    tokenizer_path: Path,
    use_local_fallback: bool = False,
) -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs: list[str] = []
    n_tokens = 0
    bytes_written = 0

    if use_local_fallback:
        logger.warning("using local .md fallback corpus (no HF stream)")
        source = iter(_gather_local_fallback())
    else:
        try:
            source = _stream_fineweb()
        except Exception as e:
            logger.warning("FineWeb stream failed (%s); falling back to local .md", e)
            source = iter(_gather_local_fallback())

    # Atomic write: SIGKILL or OOM mid-stream must not leave a half-file
    # that the next llama-dump-activations call silently consumes.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_docs = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for text in source:
            line = text.replace("\n", " ").replace("\r", " ")
            ids = tokenizer.encode(line, add_special_tokens=False)
            n_tokens += len(ids)
            f.write(line + "\n")
            n_docs += 1
            bytes_written += len(line.encode("utf-8")) + 1
            if n_tokens >= target_tokens:
                break
    import os as _os
    _os.replace(tmp_path, out_path)

    return {
        "bytes_written": bytes_written,
        "doc_count": n_docs,
        "token_count": n_tokens,
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                    help="Path to a HF tokenizer directory (default: Qwen3-1.7B).")
    ap.add_argument("--local-fallback", action="store_true",
                    help="Skip HF and use only local .md files. Bootstrap only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer not found at {args.tokenizer}.")

    stats = sample(
        target_tokens=args.target_tokens,
        out_path=args.out,
        tokenizer_path=args.tokenizer,
        use_local_fallback=args.local_fallback,
    )
    print(f"bytes_written = {stats['bytes_written']}")
    print(f"doc_count     = {stats['doc_count']}")
    print(f"token_count   = {stats['token_count']}")
    print(f"out_path      = {stats['out_path']}")


if __name__ == "__main__":
    main()
