#!/usr/bin/env python3
"""Extract TiTok VQ-8K codebook and save as standalone tensor."""
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

model_path = Path.home() / ".cache/huggingface/hub/models--yucornetto--tokenizer_titok_bl64_vq8k_imagenet/snapshots"
# Find the snapshot dir
snapshots = list(model_path.glob("*/model.safetensors"))
if not snapshots:
    raise FileNotFoundError(f"No model.safetensors in {model_path}")

state = load_file(str(snapshots[0]))
codebook = state["quantize.embedding.weight"]
logger.info("Codebook shape: %s, dtype: %s", codebook.shape, codebook.dtype)
logger.info("Codebook stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
            codebook.min(), codebook.max(), codebook.mean(), codebook.std())

# Compute pairwise distances to understand codebook structure
norms = codebook.norm(dim=1)
logger.info("Codebook norm stats: min=%.4f max=%.4f mean=%.4f", norms.min(), norms.max(), norms.mean())

# Save standalone
out_path = Path(__file__).resolve().parent / "codebooks" / "titok_vq8k_codebook.safetensors"
out_path.parent.mkdir(exist_ok=True)
save_file({"codebook": codebook}, str(out_path))
logger.info("Saved codebook to %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

# Also save decoder weights for later
decoder_keys = [k for k in state if k.startswith("decoder.")]
decoder_state = {k: v for k, v in state.items() if k.startswith("decoder.")}
decoder_path = out_path.parent / "titok_decoder.safetensors"
save_file(decoder_state, str(decoder_path))
logger.info("Saved decoder (%d tensors, %.1f MB) to %s",
            len(decoder_state), decoder_path.stat().st_size / 1024 / 1024, decoder_path)

# Save encoder too
encoder_keys = [k for k in state if k.startswith("encoder.")]
encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
encoder_path = out_path.parent / "titok_encoder.safetensors"
save_file(encoder_state, str(encoder_path))
logger.info("Saved encoder (%d tensors, %.1f MB) to %s",
            len(encoder_state), encoder_path.stat().st_size / 1024 / 1024, encoder_path)
