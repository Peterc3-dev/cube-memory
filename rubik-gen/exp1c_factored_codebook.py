#!/usr/bin/env python3
"""Rubik Gen — Experiment 1c: Factored Codebook VSA Binding.

Exp 1 and 1b showed that flat retrieval from 8192 codebook entries
(13 bits) fails because FHRR superposition of 64 bindings provides
only ~log2(D/64) bits per position.

Key insight: FACTOR the codebook. Decompose 8192 = 128 × 64.
Each token index becomes (factor_a, factor_b) where:
    token = factor_a * 64 + factor_b

Now each retrieval only needs to distinguish 128 or 64 entries
(7 or 6 bits), which is achievable at moderate D.

Architecture:
    - Position phasors: (64, D) frozen random
    - Factor-A phasors: (128, D) learned
    - Factor-B phasors: (64, D) learned
    - Binding: pos_key ⊗ factA_key ⊗ factB_key (3-way binding)
    - Retrieval: unbind pos → unbind factA → classify factB (and vice versa)
    - Loss: CE on both factor predictions
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

N_POS = 64
N_CODES = 8192
FACT_A = 128  # 8192 // 64
FACT_B = 64   # 8192 // 128


class FactoredVSABinder(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.d = d

        g = torch.Generator().manual_seed(42)
        ph = (torch.rand(N_POS, d, generator=g) * 2 - 1) * torch.pi
        self.register_buffer("pos_cos", ph.cos())
        self.register_buffer("pos_sin", ph.sin())

        self.factA_embed = nn.Parameter(torch.randn(FACT_A, d) * 0.02)
        self.factB_embed = nn.Parameter(torch.randn(FACT_B, d) * 0.02)

    def _bind_cos_sin(self, c1, s1, c2, s2):
        """Phasor multiply: (c1+is1)(c2+is2) = (c1c2-s1s2) + i(c1s2+s1c2)."""
        return c1 * c2 - s1 * s2, c1 * s2 + s1 * c2

    def encode(self, token_indices):
        """3-way bind: pos ⊗ factA ⊗ factB, then superpose."""
        B = token_indices.shape[0]
        fa_idx = token_indices // FACT_B  # (B, 64)
        fb_idx = token_indices % FACT_B   # (B, 64)

        fa_phases = self.factA_embed[fa_idx]  # (B, 64, d)
        fb_phases = self.factB_embed[fb_idx]

        fa_c, fa_s = fa_phases.cos(), fa_phases.sin()
        fb_c, fb_s = fb_phases.cos(), fb_phases.sin()

        # Bind factA ⊗ factB
        ab_c, ab_s = self._bind_cos_sin(fa_c, fa_s, fb_c, fb_s)

        # Bind pos ⊗ (factA ⊗ factB)
        bound_c, bound_s = self._bind_cos_sin(
            self.pos_cos.unsqueeze(0), self.pos_sin.unsqueeze(0),
            ab_c, ab_s,
        )

        # Superpose across positions
        sr, si = bound_c.sum(1), bound_s.sum(1)
        mag = (sr ** 2 + si ** 2).sqrt().clamp(min=1e-8)
        return sr / mag, si / mag

    def retrieve_factors(self, sr, si):
        """Unbind positions, then retrieve both factors."""
        B = sr.shape[0]

        # Unbind all positions: multiply by pos.conj()
        ur = sr.unsqueeze(1) * self.pos_cos.unsqueeze(0) + si.unsqueeze(1) * self.pos_sin.unsqueeze(0)
        ui = si.unsqueeze(1) * self.pos_cos.unsqueeze(0) - sr.unsqueeze(1) * self.pos_sin.unsqueeze(0)
        # ur, ui: (B, 64, d) — contains factA ⊗ factB + noise

        # Strategy 1: unbind factA, classify factB
        fa_c = self.factA_embed.cos()  # (128, d)
        fa_s = self.factA_embed.sin()
        fb_c = self.factB_embed.cos()  # (64, d)
        fb_s = self.factB_embed.sin()

        # For each candidate factA, unbind it and measure similarity to all factBs
        # This is O(FACT_A * FACT_B * D) per position — feasible at 128*64=8192
        # But we can be smarter: score each factor independently

        # Score factA: similarity of unbound signal to each factA
        # (ignoring factB — noisy but gives a ranking)
        B2, P, D = ur.shape
        ur_flat = ur.reshape(B2 * P, D)
        ui_flat = ui.reshape(B2 * P, D)

        logits_a = (ur_flat @ fa_c.T + ui_flat @ fa_s.T) / self.d  # (B*64, 128)

        # Score factB: unbind best factA candidate, then score factBs
        # But for training, we can also directly score factB from the raw unbound
        logits_b = (ur_flat @ fb_c.T + ui_flat @ fb_s.T) / self.d  # (B*64, 64)

        return logits_a.reshape(B2, P, FACT_A), logits_b.reshape(B2, P, FACT_B)

    def forward(self, token_indices):
        sr, si = self.encode(token_indices)
        return self.retrieve_factors(sr, si)


def run(d, n_train=1000, n_val=200, steps=3000, bs=16, lr=1e-3):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    train = torch.randint(0, N_CODES, (n_train, N_POS))
    val = torch.randint(0, N_CODES, (n_val, N_POS))
    train_a, train_b = train // FACT_B, train % FACT_B
    val_a, val_b = val // FACT_B, val % FACT_B

    model = FactoredVSABinder(d=d)
    tp = sum(p.numel() for p in model.parameters())
    logger.info("D=%d — %dK params", d, tp // 1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    history = []
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_train, (bs,), generator=rng)
        batch = train[idx]
        batch_a, batch_b = batch // FACT_B, batch % FACT_B

        logits_a, logits_b = model(batch)
        loss_a = F.cross_entropy(logits_a.reshape(-1, FACT_A), batch_a.reshape(-1))
        loss_b = F.cross_entropy(logits_b.reshape(-1, FACT_B), batch_b.reshape(-1))
        loss = loss_a + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vla, vlb = model(val)
                vl_a = F.cross_entropy(vla.reshape(-1, FACT_A), val_a.reshape(-1))
                vl_b = F.cross_entropy(vlb.reshape(-1, FACT_B), val_b.reshape(-1))

                pred_a = vla.argmax(-1)
                pred_b = vlb.argmax(-1)
                acc_a = (pred_a == val_a).float().mean().item()
                acc_b = (pred_b == val_b).float().mean().item()

                # Joint accuracy: both factors correct = correct token
                joint = ((pred_a == val_a) & (pred_b == val_b)).float().mean().item()
                exact = ((pred_a == val_a) & (pred_b == val_b)).all(1).float().mean().item()

            entry = {
                "step": step,
                "loss_a": round(loss_a.item(), 3), "loss_b": round(loss_b.item(), 3),
                "val_a": round(vl_a.item(), 3), "val_b": round(vl_b.item(), 3),
                "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
                "joint": round(joint, 4), "exact": round(exact, 4),
            }
            history.append(entry)
            logger.info(
                "  step %4d  lA=%.3f lB=%.3f  vA=%.3f vB=%.3f  accA=%.1f%% accB=%.1f%% joint=%.1f%% exact=%.1f%%",
                step, loss_a.item(), loss_b.item(), vl_a.item(), vl_b.item(),
                acc_a * 100, acc_b * 100, joint * 100, exact * 100,
            )
            model.train()

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        vla, vlb = model(val)
        pred_a, pred_b = vla.argmax(-1), vlb.argmax(-1)
        acc_a = (pred_a == val_a).float().mean().item()
        acc_b = (pred_b == val_b).float().mean().item()
        joint = ((pred_a == val_a) & (pred_b == val_b)).float().mean().item()
        exact = ((pred_a == val_a) & (pred_b == val_b)).all(1).float().mean().item()

    bits_a = torch.log2(torch.tensor(d / N_POS)).item()

    return {
        "d": d, "params": tp, "elapsed_s": round(elapsed, 1),
        "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
        "joint_acc": round(joint, 4), "exact": round(exact, 4),
        "bits_per_pos": round(bits_a, 1),
        "bits_needed_a": round(torch.log2(torch.tensor(float(FACT_A))).item(), 1),
        "bits_needed_b": round(torch.log2(torch.tensor(float(FACT_B))).item(), 1),
        "history": history,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = {}
    for d in [512, 1024, 2048, 4096]:
        logger.info("=" * 50)
        bits = torch.log2(torch.tensor(d / N_POS)).item()
        logger.info("Factored VSA: D=%d (%.1f bits/pos, need A:%.1f B:%.1f)",
                     d, bits,
                     torch.log2(torch.tensor(float(FACT_A))).item(),
                     torch.log2(torch.tensor(float(FACT_B))).item())
        logger.info("=" * 50)
        r = run(d)
        results[f"D={d}"] = r
        logger.info("D=%d DONE — accA=%.1f%% accB=%.1f%% joint=%.1f%% (%.0fs)\n",
                     d, r["acc_a"] * 100, r["acc_b"] * 100, r["joint_acc"] * 100, r["elapsed_s"])

    logger.info("=" * 50)
    logger.info("FACTORED CODEBOOK SUMMARY")
    logger.info("%-6s %-8s %-8s %-8s %-8s %-10s", "D", "AccA%", "AccB%", "Joint%", "Exact%", "Bits(have/need)")
    for k, r in results.items():
        logger.info("%-6s %-8.1f %-8.1f %-8.1f %-8.1f %.1f / (%.1f+%.1f)",
                     k, r["acc_a"] * 100, r["acc_b"] * 100,
                     r["joint_acc"] * 100, r["exact"] * 100,
                     r["bits_per_pos"], r["bits_needed_a"], r["bits_needed_b"])

    out = Path("/home/raz/projects/cube-memory/rubik-gen/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "exp1c_factored.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out / "exp1c_factored.json")


if __name__ == "__main__":
    main()
