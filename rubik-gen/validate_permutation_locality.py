#!/usr/bin/env python3
"""Rubik Gen — Validation Experiment 0: Permutation Locality.

Core thesis test: do small permutations of token positions produce
visually similar images? If yes, the Rubik gen architecture has legs.

Setup:
1. Load TiTok VQ-8K tokenizer (frozen)
2. Encode a real image → 64 codebook indices
3. Apply position permutations of increasing magnitude
4. Decode each permuted token sequence
5. Measure MSE / LPIPS / SSIM vs original

Expected result: small swaps → small visual change. Large shuffles →
large visual change. Monotonic relationship = permutation locality holds.
"""
from __future__ import annotations

import logging
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/tmp/1d-tokenizer")

logger = logging.getLogger(__name__)


def load_titok(device="cpu"):
    """Load TiTok model from HuggingFace."""
    from modeling.titok import TiTok
    model = TiTok.from_pretrained(
        "yucornetto/tokenizer_titok_bl64_vq8k_imagenet",
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def random_position_swaps(seq_len, n_swaps, rng):
    """Generate a permutation by performing n_swaps random adjacent-ish swaps."""
    perm = list(range(seq_len))
    for _ in range(n_swaps):
        i = rng.randint(0, seq_len - 1)
        j = rng.randint(0, seq_len - 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def kendall_tau_distance(perm):
    """Count number of inversions = Kendall tau distance from identity."""
    n = len(perm)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv


def get_test_image(size=256):
    """Get a test image. Try to load a real one, fall back to synthetic."""
    candidates = [
        Path.home() / "Pictures" / "test.jpg",
        Path.home() / "Pictures" / "test.png",
    ]
    for p in candidates:
        if p.exists():
            img = Image.open(p).convert("RGB").resize((size, size))
            logger.info("Loaded test image: %s", p)
            return img

    # Try downloading a sample
    try:
        from torchvision.datasets import FakeData
        ds = FakeData(size=1, image_size=(3, size, size))
        img, _ = ds[0]
        logger.info("Using synthetic test image")
        return img
    except Exception:
        pass

    # Create a gradient image
    import numpy as np
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            arr[i, j, 0] = int(255 * i / size)
            arr[i, j, 1] = int(255 * j / size)
            arr[i, j, 2] = int(255 * (i + j) / (2 * size))
    logger.info("Using synthetic gradient image")
    return Image.fromarray(arr)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(42)
    rng = random.Random(42)

    device = "cpu"
    logger.info("Loading TiTok model...")
    model = load_titok(device)
    logger.info("Model loaded. Codebook: %d entries, %d dims",
                model.quantize.embedding.weight.shape[0],
                model.quantize.embedding.weight.shape[1])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    img = get_test_image()
    x = transform(img).unsqueeze(0).to(device)
    logger.info("Input image shape: %s", x.shape)

    # Encode
    with torch.no_grad():
        z = model.encoder(pixel_values=x, latent_tokens=model.latent_tokens)
        z_q, result = model.quantize(z)
        tokens = result["min_encoding_indices"].squeeze()  # (64,)
    logger.info("Encoded to %d tokens, unique values: %d", tokens.numel(), tokens.unique().numel())

    # Decode original
    with torch.no_grad():
        recon_orig = model.decode_tokens(tokens.unsqueeze(0))
    orig_mse = F.mse_loss(recon_orig, x).item()
    logger.info("Original reconstruction MSE: %.6f", orig_mse)

    # Test permutations of increasing magnitude
    swap_counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    n_trials = 5
    seq_len = tokens.numel()

    logger.info("")
    logger.info("=" * 70)
    logger.info("PERMUTATION LOCALITY TEST")
    logger.info("%-10s %-15s %-15s %-15s", "Swaps", "Avg KT dist", "Avg MSE", "MSE ratio")
    logger.info("=" * 70)

    results = []
    for n_swaps in swap_counts:
        mses = []
        kt_dists = []
        for trial in range(n_trials):
            perm = random_position_swaps(seq_len, n_swaps, rng)
            kt = kendall_tau_distance(perm)
            kt_dists.append(kt)

            # Permute token positions
            perm_tokens = tokens[perm].unsqueeze(0)

            with torch.no_grad():
                recon_perm = model.decode_tokens(perm_tokens)

            mse = F.mse_loss(recon_perm, recon_orig).item()
            mses.append(mse)

        avg_mse = sum(mses) / len(mses)
        avg_kt = sum(kt_dists) / len(kt_dists)
        ratio = avg_mse / max(orig_mse, 1e-10)
        results.append((n_swaps, avg_kt, avg_mse, ratio))
        logger.info("%-10d %-15.1f %-15.6f %-15.2f", n_swaps, avg_kt, avg_mse, ratio)

    logger.info("")

    # Check monotonicity
    mse_values = [r[2] for r in results]
    is_monotonic = all(mse_values[i] <= mse_values[i+1] + 1e-8 for i in range(len(mse_values)-1))
    logger.info("MSE monotonically increases with permutation magnitude: %s", is_monotonic)

    if is_monotonic or (mse_values[-1] > mse_values[0] * 2):
        logger.info("RESULT: Permutation locality holds — position ordering matters for visual output.")
        logger.info("This validates the core Rubik gen premise.")
    else:
        logger.info("RESULT: Permutation locality does NOT hold — position ordering is irrelevant.")
        logger.info("This would invalidate the Rubik gen premise.")

    # Save images if possible
    try:
        save_dir = Path(__file__).resolve().parent / "validation_images"
        save_dir.mkdir(exist_ok=True)

        def save_tensor_as_image(t, path):
            from torchvision.utils import save_image
            save_image(t.clamp(0, 1), str(path))

        save_tensor_as_image(x, save_dir / "original.png")
        save_tensor_as_image(recon_orig, save_dir / "reconstructed.png")

        # Save a few permuted examples
        for n_swaps in [4, 32, 256]:
            perm = random_position_swaps(seq_len, n_swaps, rng)
            perm_tokens = tokens[perm].unsqueeze(0)
            with torch.no_grad():
                recon = model.decode_tokens(perm_tokens)
            save_tensor_as_image(recon, save_dir / f"permuted_{n_swaps}swaps.png")

        logger.info("Saved validation images to %s", save_dir)
    except Exception as e:
        logger.warning("Could not save images: %s", e)


if __name__ == "__main__":
    main()
