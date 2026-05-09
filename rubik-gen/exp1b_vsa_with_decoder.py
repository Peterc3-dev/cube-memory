#!/usr/bin/env python3
"""Rubik Gen — Experiment 1b: VSA Binding + Learned Decoder.

Pure VSA retrieval is capacity-limited to ~log2(D/N_pos) bits per position.
With D=4096 and 64 positions, that's ~6 bits — insufficient for 8192 codebook
entries (13 bits needed).

This experiment adds a small MLP decoder after unbinding:
    unbind(state, pos_key) → noisy signal → MLP → logits over codebook

The VSA binding still provides the algebraic structure (permutation-equivariant),
but the MLP learns to extract the signal from superposition noise.

Key: the VSA state is still the "representation" — permutation ops act on it.
The decoder is position-shared (same MLP for all 64 positions) so total params
stay small.
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


class VSAWithDecoder(nn.Module):

    def __init__(self, d_vsa, d_hidden=512, n_codes=N_CODES):
        super().__init__()
        self.d_vsa = d_vsa
        self.n_codes = n_codes

        g = torch.Generator().manual_seed(42)
        ph = (torch.rand(N_POS, d_vsa, generator=g) * 2 - 1) * torch.pi
        self.register_buffer("pos_cos", ph.cos())
        self.register_buffer("pos_sin", ph.sin())

        self.code_embed = nn.Parameter(torch.randn(n_codes, d_vsa) * 0.02)

        self.decoder = nn.Sequential(
            nn.Linear(2 * d_vsa, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_codes),
        )

    def encode(self, token_indices):
        """Bind and superpose → state (real, imag)."""
        cp = self.code_embed[token_indices]
        cc, cs = cp.cos(), cp.sin()
        br = self.pos_cos.unsqueeze(0) * cc - self.pos_sin.unsqueeze(0) * cs
        bi = self.pos_cos.unsqueeze(0) * cs + self.pos_sin.unsqueeze(0) * cc
        sr, si = br.sum(1), bi.sum(1)
        mag = (sr ** 2 + si ** 2).sqrt().clamp(min=1e-8)
        return sr / mag, si / mag

    def decode_all(self, sr, si):
        """Unbind all positions, pass through decoder MLP."""
        B = sr.shape[0]
        # Unbind all positions at once
        # sr: (B, d), pos_cos: (N_POS, d) → ur: (B, N_POS, d)
        ur = sr.unsqueeze(1) * self.pos_cos.unsqueeze(0) + si.unsqueeze(1) * self.pos_sin.unsqueeze(0)
        ui = si.unsqueeze(1) * self.pos_cos.unsqueeze(0) - sr.unsqueeze(1) * self.pos_sin.unsqueeze(0)

        # Concat real+imag and decode
        x = torch.cat([ur, ui], dim=-1)  # (B, N_POS, 2*d_vsa)
        logits = self.decoder(x.reshape(B * N_POS, -1))
        return logits.reshape(B, N_POS, self.n_codes)

    def forward(self, token_indices):
        sr, si = self.encode(token_indices)
        return self.decode_all(sr, si)


def run(d_vsa, d_hidden=512, n_train=1000, n_val=200, steps=2000, bs=16, lr=1e-3):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)

    train = torch.randint(0, N_CODES, (n_train, N_POS))
    val = torch.randint(0, N_CODES, (n_val, N_POS))

    model = VSAWithDecoder(d_vsa=d_vsa, d_hidden=d_hidden)
    tp = sum(p.numel() for p in model.parameters())
    tp_dec = sum(p.numel() for p in model.decoder.parameters())
    tp_embed = model.code_embed.numel()
    logger.info("D=%d, hidden=%d — total %dK params (embed: %dK, decoder: %dK)",
                d_vsa, d_hidden, tp // 1000, tp_embed // 1000, tp_dec // 1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    history = []
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_train, (bs,), generator=rng)
        batch = train[idx]
        logits = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, N_CODES), batch.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = model(val)
                vloss = F.cross_entropy(vl.reshape(-1, N_CODES), val.reshape(-1))
                vp = vl.argmax(-1)
                acc = (vp == val).float().mean().item()
                ex = (vp == val).all(1).float().mean().item()
                t5 = (vl.reshape(-1, N_CODES).topk(5, -1).indices
                      .eq(val.reshape(-1).unsqueeze(-1)).any(-1).float().mean().item())
            history.append({"step": step, "loss": round(loss.item(), 3),
                           "val": round(vloss.item(), 3), "acc": round(acc, 4),
                           "top5": round(t5, 4), "exact": round(ex, 4)})
            logger.info("  step %4d  loss=%.3f  val=%.3f  acc=%.2f%%  top5=%.2f%%  exact=%.2f%%",
                        step, loss.item(), vloss.item(), acc * 100, t5 * 100, ex * 100)
            model.train()

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        vl = model(val)
        vp = vl.argmax(-1)
        acc = (vp == val).float().mean().item()
        ex = (vp == val).all(1).float().mean().item()
        t5 = (vl.reshape(-1, N_CODES).topk(5, -1).indices
              .eq(val.reshape(-1).unsqueeze(-1)).any(-1).float().mean().item())

    return {
        "d_vsa": d_vsa, "d_hidden": d_hidden,
        "params_total": tp, "params_embed": tp_embed, "params_decoder": tp_dec,
        "elapsed_s": round(elapsed, 1),
        "acc": round(acc, 4), "top5": round(t5, 4), "exact": round(ex, 4),
        "history": history,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    configs = [
        {"d_vsa": 256, "d_hidden": 256},
        {"d_vsa": 512, "d_hidden": 512},
        {"d_vsa": 1024, "d_hidden": 512},
        {"d_vsa": 2048, "d_hidden": 512},
    ]

    results = {}
    for cfg in configs:
        d, h = cfg["d_vsa"], cfg["d_hidden"]
        logger.info("=" * 50)
        logger.info("VSA+Decoder: D=%d, hidden=%d", d, h)
        logger.info("=" * 50)
        r = run(**cfg)
        results[f"D={d}_H={h}"] = r
        logger.info("DONE — acc=%.1f%% top5=%.1f%% (%.0fs)\n",
                     r["acc"] * 100, r["top5"] * 100, r["elapsed_s"])

    logger.info("=" * 50)
    logger.info("COMPARISON: Pure VSA vs VSA+Decoder")
    logger.info("%-15s %-8s %-8s %-8s", "Config", "Acc%", "Top5%", "Params")
    for k, r in results.items():
        logger.info("%-15s %-8.1f %-8.1f %dK", k, r["acc"] * 100, r["top5"] * 100, r["params_total"] // 1000)

    out_dir = Path("/home/raz/projects/cube-memory/rubik-gen/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp1b_vsa_decoder.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_dir / "exp1b_vsa_decoder.json")


if __name__ == "__main__":
    main()
