#!/usr/bin/env python3
"""Fetch and inspect a pre-trained VQ codebook for Rubik gen.

Downloads the TiTok VQ-8K tokenizer from HuggingFace, extracts the
codebook embedding, and saves it as a standalone tensor for downstream
use in the permutation-group experiment.

Fallback: LlamaGen ds16 (16K codebook, 8-dim embeddings).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def try_titok():
    """Try to download and extract TiTok VQ-8K codebook."""
    try:
        from huggingface_hub import hf_hub_download
        repo = "yucornetto/tokenizer_titok_bl64_vq8k_imagenet"
        # Try to find the model file
        files = ["model.safetensors", "pytorch_model.bin"]
        for fname in files:
            try:
                path = hf_hub_download(repo_id=repo, filename=fname)
                logger.info("Downloaded %s from %s", fname, repo)
                return path, repo
            except Exception:
                continue
        # Try listing the repo
        from huggingface_hub import list_repo_files
        all_files = list(list_repo_files(repo))
        logger.info("Files in %s: %s", repo, all_files)
        # Look for any safetensors or .bin
        for f in all_files:
            if f.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                path = hf_hub_download(repo_id=repo, filename=f)
                logger.info("Downloaded %s", f)
                return path, repo
    except Exception as e:
        logger.warning("TiTok download failed: %s", e)
    return None, None


def try_llamagen():
    """Try LlamaGen ds16 VQ tokenizer."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        repo = "FoundationVision/vq-ds16-c2i"
        all_files = list(list_repo_files(repo))
        logger.info("Files in %s: %s", repo, all_files)
        for f in all_files:
            if f.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                path = hf_hub_download(repo_id=repo, filename=f)
                logger.info("Downloaded %s from %s", f, repo)
                return path, repo
    except Exception as e:
        logger.warning("LlamaGen download failed: %s", e)
    return None, None


def extract_codebook(model_path: str, repo: str):
    """Load model weights and find the codebook tensor."""
    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    logger.info("Loaded %d tensors from %s", len(state), model_path)

    # Search for codebook-like tensors
    candidates = []
    for k, v in state.items():
        kl = k.lower()
        if any(s in kl for s in ["codebook", "embed", "quantize", "vq", "dictionary"]):
            candidates.append((k, v.shape, v.dtype))
            logger.info("  Candidate: %s  shape=%s  dtype=%s", k, v.shape, v.dtype)

    if not candidates:
        logger.info("No obvious codebook keys. Listing all tensors with 2D shape:")
        for k, v in state.items():
            if v.ndim == 2:
                logger.info("  %s  shape=%s", k, v.shape)

    return state


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(__file__).resolve().parent / "codebooks"
    out_dir.mkdir(exist_ok=True)

    # Try TiTok first
    path, repo = try_titok()
    if path is None:
        path, repo = try_llamagen()
    if path is None:
        logger.error("Could not download any tokenizer. Check network/auth.")
        return

    state = extract_codebook(path, repo)

    # Save full state dict keys for inspection
    keys_file = out_dir / "model_keys.txt"
    with open(keys_file, "w") as f:
        for k, v in state.items():
            f.write(f"{k}\t{list(v.shape)}\t{v.dtype}\n")
    logger.info("Saved key listing to %s", keys_file)


if __name__ == "__main__":
    main()
