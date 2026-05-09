#!/usr/bin/env python3
"""Rubik Gen — Experiment 1: VSA Token Binding Capacity Test.

Can FHRR binding + learned codebook hypervectors reconstruct token
sequences after superposition?

Setup:
    - N_POSITIONS=64 frozen random position phasors
    - N_CODEBOOK=8192 learned content phasors
    - bind(pos_i, content_j) → superpose all 64 → state
    - unbind each position → classify against all codebook entries
    - Cross-entropy loss on retrieved vs ground truth index

Theory: ~log2(D/N_POSITIONS) bits per retrieval. Codebook needs 13 bits.
Tests D=512,1024,2048,4096 to find the capacity knee.

Optimized: real-arithmetic similarity (2 BLAS matmuls instead of complex einsum).
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

logger = logging.getLogger(__name__)

N_POSITIONS = 64
N_CODEBOOK = 8192


class VSATokenBinder(nn.Module):

    def __init__(self, d, n_pos=N_POSITIONS, n_codes=N_CODEBOOK):
        super().__init__()
        self.d = d
        self.n_pos = n_pos
        self.n_codes = n_codes

        g = torch.Generator().manual_seed(42)
        pos_phases = (torch.rand(n_pos, d, generator=g) * 2 - 1) * torch.pi
        self.register_buffer("pos_cos", pos_phases.cos())
        self.register_buffer("pos_sin", pos_phases.sin())

        self.code_embed = nn.Parameter(torch.randn(n_codes, d) * 0.02)

    def encode_image(self, token_indices):
        """Bind position phasors with content phasors and superpose."""
        B = token_indices.shape[0]
        code_phases = self.code_embed[token_indices]  # (B, 64, d)
        code_cos = code_phases.cos()
        code_sin = code_phases.sin()

        # Binding: multiply phasors = (pos_cos + i*pos_sin)(code_cos + i*code_sin)
        # Real part: pos_cos*code_cos - pos_sin*code_sin
        # Imag part: pos_cos*code_sin + pos_sin*code_cos
        bound_real = self.pos_cos.unsqueeze(0) * code_cos - self.pos_sin.unsqueeze(0) * code_sin
        bound_imag = self.pos_cos.unsqueeze(0) * code_sin + self.pos_sin.unsqueeze(0) * code_cos

        # Superpose: sum across positions
        state_real = bound_real.sum(dim=1)  # (B, d)
        state_imag = bound_imag.sum(dim=1)  # (B, d)

        # Unitize: normalize to unit magnitude
        mag = (state_real ** 2 + state_imag ** 2).sqrt().clamp(min=1e-8)
        state_real = state_real / mag
        state_imag = state_imag / mag

        return state_real, state_imag

    def retrieve_all_positions(self, state_real, state_imag):
        """Unbind all positions and compute similarity to all codes."""
        # Unbind: multiply by pos.conj() = (pos_cos - i*pos_sin)
        # unbound_real = state_real*pos_cos + state_imag*pos_sin
        # unbound_imag = state_imag*pos_cos - state_real*pos_sin
        # Shape: (B, 1, d) * (1, n_pos, d) → (B, n_pos, d)
        sr = state_real.unsqueeze(1)  # (B, 1, d)
        si = state_imag.unsqueeze(1)
        pc = self.pos_cos.unsqueeze(0)  # (1, n_pos, d)
        ps = self.pos_sin.unsqueeze(0)

        unbound_real = sr * pc + si * ps  # (B, n_pos, d)
        unbound_imag = si * pc - sr * ps  # (B, n_pos, d)

        # Similarity to all codes: (unbound · code.conj()).real / D
        # = (unbound_real * code_cos + unbound_imag * code_sin).sum(d) / D
        code_cos = self.code_embed.cos()  # (n_codes, d)
        code_sin = self.code_embed.sin()  # (n_codes, d)

        # Use matmul: (B*n_pos, d) @ (d, n_codes)
        B, P, D = unbound_real.shape
        ur_flat = unbound_real.reshape(B * P, D)
        ui_flat = unbound_imag.reshape(B * P, D)

        sims = (ur_flat @ code_cos.T + ui_flat @ code_sin.T) / self.d
        return sims.reshape(B, P, self.n_codes)


def run_experiment(d_vsa, n_train=1000, n_val=200, steps=5000, bs=32, lr=1e-3):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    train_tokens = torch.randint(0, N_CODEBOOK, (n_train, N_POSITIONS))
    val_tokens = torch.randint(0, N_CODEBOOK, (n_val, N_POSITIONS))

    binder = VSATokenBinder(d=d_vsa)
    tp = sum(p.numel() for p in binder.parameters())
    logger.info("D=%d — %d params (%.1f MB)", d_vsa, tp, tp * 4 / 1024 / 1024)

    optimizer = torch.optim.AdamW(binder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    history = []
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_train, (bs,), generator=rng)
        batch = train_tokens[idx]

        sr, si = binder.encode_image(batch)
        logits = binder.retrieve_all_positions(sr, si)
        loss = F.cross_entropy(logits.reshape(-1, N_CODEBOOK), batch.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == steps - 1:
            binder.eval()
            with torch.no_grad():
                val_sr, val_si = binder.encode_image(val_tokens)
                val_logits = binder.retrieve_all_positions(val_sr, val_si)
                val_loss = F.cross_entropy(val_logits.reshape(-1, N_CODEBOOK), val_tokens.reshape(-1))
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_tokens).float().mean().item()
                exact = (val_preds == val_tokens).all(dim=1).float().mean().item()
                top5 = (val_logits.reshape(-1, N_CODEBOOK)
                        .topk(5, dim=-1).indices
                        .eq(val_tokens.reshape(-1).unsqueeze(-1))
                        .any(dim=-1).float().mean().item())

            entry = {
                "step": step, "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss.item(), 4), "val_acc": round(val_acc, 4),
                "val_top5": round(top5, 4), "exact_match": round(exact, 4),
            }
            history.append(entry)
            logger.info("  step %4d  loss=%.3f  val=%.3f  acc=%.2f%%  top5=%.2f%%  exact=%.2f%%",
                        step, loss.item(), val_loss.item(), val_acc * 100, top5 * 100, exact * 100)
            binder.train()

    elapsed = time.time() - t0

    binder.eval()
    with torch.no_grad():
        val_sr, val_si = binder.encode_image(val_tokens)
        val_logits = binder.retrieve_all_positions(val_sr, val_si)
        val_preds = val_logits.argmax(dim=-1)
        val_acc = (val_preds == val_tokens).float().mean().item()
        exact = (val_preds == val_tokens).all(dim=1).float().mean().item()
        top5 = (val_logits.reshape(-1, N_CODEBOOK)
                .topk(5, dim=-1).indices
                .eq(val_tokens.reshape(-1).unsqueeze(-1))
                .any(dim=-1).float().mean().item())

    return {
        "d_vsa": d_vsa,
        "n_train": n_train, "n_val": n_val,
        "n_params": tp, "elapsed_s": round(elapsed, 1),
        "final_val_acc": round(val_acc, 4),
        "final_top5": round(top5, 4),
        "final_exact_match": round(exact, 4),
        "theory_bits_per_pos": round(torch.log2(torch.tensor(d_vsa / N_POSITIONS)).item(), 2),
        "needed_bits": round(torch.log2(torch.tensor(float(N_CODEBOOK))).item(), 2),
        "history": history,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = {}
    for d in [4096, 8192, 16384]:
        logger.info("=" * 60)
        logger.info("D=%d (theory: %.1f bits/pos, need %.1f)",
                     d, torch.log2(torch.tensor(d / N_POSITIONS)).item(),
                     torch.log2(torch.tensor(float(N_CODEBOOK))).item())
        logger.info("=" * 60)
        r = run_experiment(d_vsa=d)
        results[f"D={d}"] = r
        logger.info("D=%d — acc=%.1f%% top5=%.1f%% exact=%.1f%% (%.0fs)\n",
                     d, r["final_val_acc"] * 100, r["final_top5"] * 100,
                     r["final_exact_match"] * 100, r["elapsed_s"])

    logger.info("=" * 60)
    logger.info("CAPACITY SUMMARY")
    logger.info("%-6s %-8s %-8s %-8s %-8s %-10s", "D", "Params", "Acc%", "Top5%", "Exact%", "Bits(have/need)")
    for k, r in results.items():
        logger.info("%-6s %-8s %-8.1f %-8.1f %-8.1f %.1f / %.1f",
                     k, f"{r['n_params']//1000}K", r["final_val_acc"] * 100,
                     r["final_top5"] * 100, r["final_exact_match"] * 100,
                     r["theory_bits_per_pos"], r["needed_bits"])

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp1_vsa_capacity.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_dir / "exp1_vsa_capacity.json")


if __name__ == "__main__":
    main()
