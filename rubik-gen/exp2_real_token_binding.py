#!/usr/bin/env python3
"""Rubik Gen — Experiment 2: VSA Binding on Real TiTok Tokens.

After exp1 establishes VSA capacity on random tokens, this tests
whether the approach works with real image tokens from TiTok.

Real tokens differ from random in two key ways:
1. Token distributions are non-uniform (some codebook entries are hot)
2. Tokens have spatial correlations (nearby positions use similar codes)

These correlations might help (structure to exploit) or hurt
(concentrated codebook use → higher collision rate in VSA).

Uses Imagenette (10 ImageNet classes, ~13k images) encoded by
frozen TiTok VQ-8K → 64 codebook indices per image.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/tmp/1d-tokenizer")

logger = logging.getLogger(__name__)

N_POSITIONS = 64
N_CODEBOOK = 8192


def _to_phasor(x):
    return torch.complex(x.cos(), x.sin())


def _unitize(z):
    return z / z.abs().clamp(min=1e-8)


class VSATokenBinder(nn.Module):

    def __init__(self, d, n_pos=N_POSITIONS, n_codes=N_CODEBOOK):
        super().__init__()
        self.d = d
        self.n_pos = n_pos
        self.n_codes = n_codes

        g = torch.Generator().manual_seed(42)
        pos_phases = (torch.rand(n_pos, d, generator=g) * 2 - 1) * torch.pi
        self.register_buffer("pos_keys", _to_phasor(pos_phases))
        self.code_embed = nn.Parameter(torch.randn(n_codes, d) * 0.02)

    def encode_image(self, token_indices):
        B = token_indices.shape[0]
        code_hvs = _to_phasor(self.code_embed[token_indices])
        bound = self.pos_keys.unsqueeze(0) * code_hvs
        state = _unitize(bound.sum(dim=1))
        return state

    def retrieve_all_positions(self, state):
        code_phasors = _to_phasor(self.code_embed)
        unbound = state.unsqueeze(1) * self.pos_keys.conj().unsqueeze(0)
        sims = torch.einsum("bpd,cd->bpc", unbound, code_phasors.conj()).real / self.d
        return sims


def encode_dataset(titok_model, image_dir, max_images=2000):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    image_dir = Path(image_dir)
    img_paths = sorted(image_dir.rglob("*.JPEG"))[:max_images]
    if not img_paths:
        img_paths = sorted(image_dir.rglob("*.jpg"))[:max_images]
    if not img_paths:
        img_paths = sorted(image_dir.rglob("*.png"))[:max_images]

    logger.info("Found %d images in %s", len(img_paths), image_dir)

    all_tokens = []
    batch_imgs = []
    batch_size = 32

    for i, p in enumerate(img_paths):
        try:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(transform(img))
        except Exception:
            continue

        if len(batch_imgs) == batch_size or i == len(img_paths) - 1:
            if not batch_imgs:
                continue
            x = torch.stack(batch_imgs)
            with torch.no_grad():
                z = titok_model.encoder(pixel_values=x, latent_tokens=titok_model.latent_tokens.expand(len(x), -1, -1))
                _, result = titok_model.quantize(z)
                toks = result["min_encoding_indices"].reshape(len(x), N_POSITIONS)
            all_tokens.append(toks)
            batch_imgs = []

        if (i + 1) % 500 == 0:
            logger.info("  encoded %d / %d", i + 1, len(img_paths))

    tokens = torch.cat(all_tokens)
    logger.info("Encoded %d images → token tensor %s", len(tokens), tokens.shape)
    return tokens


def analyze_token_distribution(tokens):
    flat = tokens.reshape(-1)
    n_total = flat.numel()
    counts = torch.bincount(flat, minlength=N_CODEBOOK)
    n_used = (counts > 0).sum().item()
    top10 = counts.topk(10)

    logger.info("Token distribution:")
    logger.info("  Unique codes used: %d / %d (%.1f%%)", n_used, N_CODEBOOK, n_used / N_CODEBOOK * 100)
    logger.info("  Top 10 codes: %s", list(zip(top10.indices.tolist(), top10.values.tolist())))
    logger.info("  Entropy: %.2f bits (max %.2f)",
                -(counts[counts > 0].float() / n_total * (counts[counts > 0].float() / n_total).log2()).sum().item(),
                torch.log2(torch.tensor(float(N_CODEBOOK))).item())

    # Spatial correlation: do adjacent positions tend to use similar codes?
    same_adj = 0
    total_adj = 0
    for i in range(N_POSITIONS - 1):
        same_adj += (tokens[:, i] == tokens[:, i + 1]).sum().item()
        total_adj += tokens.shape[0]
    logger.info("  Adjacent position same-code rate: %.4f (random baseline: %.4f)",
                same_adj / total_adj, 1.0 / N_CODEBOOK)

    return {"n_unique": n_used, "top10_codes": top10.indices.tolist(), "top10_counts": top10.values.tolist()}


def train_and_eval(d_vsa, train_tokens, val_tokens, steps=5000, bs=32, lr=1e-3):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    binder = VSATokenBinder(d=d_vsa)
    tp = sum(p.numel() for p in binder.parameters())
    logger.info("D=%d — %d params (%.1f MB)", d_vsa, tp, tp * 4 / 1024 / 1024)

    optimizer = torch.optim.AdamW(binder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    history = []
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, len(train_tokens), (bs,), generator=rng)
        batch = train_tokens[idx]

        state = binder.encode_image(batch)
        logits = binder.retrieve_all_positions(state)
        loss = F.cross_entropy(logits.reshape(-1, N_CODEBOOK), batch.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == steps - 1:
            binder.eval()
            with torch.no_grad():
                # Eval in chunks to avoid OOM
                val_losses, val_correct, val_top5_correct, val_total = [], 0, 0, 0
                val_exact = 0
                chunk = 50
                for vi in range(0, len(val_tokens), chunk):
                    vb = val_tokens[vi:vi + chunk]
                    vs = binder.encode_image(vb)
                    vl = binder.retrieve_all_positions(vs)
                    val_losses.append(F.cross_entropy(vl.reshape(-1, N_CODEBOOK), vb.reshape(-1)).item())
                    vp = vl.argmax(dim=-1)
                    val_correct += (vp == vb).sum().item()
                    val_top5_correct += (vl.reshape(-1, N_CODEBOOK).topk(5, dim=-1).indices
                                         .eq(vb.reshape(-1).unsqueeze(-1)).any(dim=-1).sum().item())
                    val_total += vb.numel()
                    val_exact += (vp == vb).all(dim=1).sum().item()

                val_loss = sum(val_losses) / len(val_losses)
                val_acc = val_correct / val_total
                top5 = val_top5_correct / val_total
                exact = val_exact / len(val_tokens)

            entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_top5": round(top5, 4),
                "exact_match": round(exact, 4),
            }
            history.append(entry)
            logger.info("  step %5d  loss=%.3f  val=%.3f  acc=%.2f%%  top5=%.2f%%  exact=%.2f%%",
                        step, loss.item(), val_loss, val_acc * 100, top5 * 100, exact * 100)
            binder.train()

    elapsed = time.time() - t0

    return {
        "d_vsa": d_vsa,
        "n_train": len(train_tokens),
        "n_val": len(val_tokens),
        "n_params": tp,
        "elapsed_s": round(elapsed, 1),
        "final_val_acc": round(val_acc, 4),
        "final_top5": round(top5, 4),
        "final_exact_match": round(exact, 4),
        "history": history,
    }, binder


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_path = Path(__file__).resolve().parent / "cached_tokens" / "imagenette_tokens.pt"

    if cache_path.exists():
        logger.info("Loading cached tokens from %s", cache_path)
        tokens = torch.load(cache_path, weights_only=True)
    else:
        logger.info("Loading TiTok for encoding...")
        from modeling.titok import TiTok
        titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_bl64_vq8k_imagenet")
        titok.eval()
        for p in titok.parameters():
            p.requires_grad_(False)

        dataset_dir = Path("/tmp/imagenette2-320")
        if not dataset_dir.exists():
            logger.error("Imagenette not found at %s — download first", dataset_dir)
            sys.exit(1)

        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"

        logger.info("Encoding training images...")
        train_tokens_raw = encode_dataset(titok, train_dir, max_images=5000)
        logger.info("Encoding validation images...")
        val_tokens_raw = encode_dataset(titok, val_dir, max_images=1000)

        tokens = torch.cat([train_tokens_raw, val_tokens_raw])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)
        logger.info("Cached %d token sequences to %s", len(tokens), cache_path)

    # Split
    n = len(tokens)
    n_val = max(100, int(n * 0.15))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_tokens = tokens[perm[:n_val]]
    train_tokens = tokens[perm[n_val:]]
    logger.info("Train: %d, Val: %d", len(train_tokens), len(val_tokens))

    # Analyze distribution
    dist_info = analyze_token_distribution(tokens)

    # Pick best D from exp1, or test the same sweep
    results = {}

    # Use D that exp1 found best, but also test a smaller one for comparison
    for d in [4096, 8192]:
        logger.info("=" * 60)
        logger.info("REAL TOKENS: D=%d", d)
        logger.info("=" * 60)
        r, binder = train_and_eval(d, train_tokens, val_tokens, steps=5000)
        results[f"D={d}"] = r
        logger.info("D=%d DONE — acc=%.1f%% top5=%.1f%% exact=%.1f%% (%.0fs)",
                     d, r["final_val_acc"] * 100, r["final_top5"] * 100,
                     r["final_exact_match"] * 100, r["elapsed_s"])

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("REAL TOKEN RESULTS")
    logger.info("%-8s %-10s %-10s %-10s", "D", "Acc%", "Top5%", "Exact%")
    for k, r in results.items():
        logger.info("%-8s %-10.1f %-10.1f %-10.1f", k, r["final_val_acc"] * 100,
                     r["final_top5"] * 100, r["final_exact_match"] * 100)

    results["token_distribution"] = dist_info

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp2_real_tokens.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_dir / "exp2_real_tokens.json")


if __name__ == "__main__":
    main()
