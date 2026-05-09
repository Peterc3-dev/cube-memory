#!/usr/bin/env python3
"""Rubik Gen — Experiment 3: VSA Permutation Representation.

Exps 1-2 showed VSA can't retrieve individual token identities from
superposition (8192-way classification from 64 bindings fails).

Key reframe: don't store TOKENS in superposition — store PERMUTATIONS.

A permutation σ of 64 positions is encoded as:
    perm_hv = Σ_i bind(pos_source_i, pos_dest_σ(i))

Decoding: unbind pos_source_i → classify among 64 positions → get σ(i).
This is only 64-way classification (6 bits), not 8192-way (13 bits).

Crucially, for SMALL permutations (few swaps), most bindings are
identity: bind(pos_i, pos_i). These contribute no cross-talk.
Only k non-identity bindings add noise, so effective superposition
is k, not 64. This should make small permutations easy to represent.

Tests accuracy vs number of swaps (k=1,2,4,8,16,32,64) across D values.
This directly validates whether VSA + permutation-locality = viable.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

N_POS = 64


def make_position_phasors(n_pos, d, seed=42):
    g = torch.Generator().manual_seed(seed)
    ph = (torch.rand(n_pos, d, generator=g) * 2 - 1) * torch.pi
    return ph.cos(), ph.sin()


def random_permutation_with_k_swaps(n, k, rng=None):
    """Generate a permutation that is exactly k transpositions from identity."""
    perm = torch.arange(n)
    positions = torch.randperm(n, generator=rng)[:2 * k].reshape(k, 2)
    for a, b in positions:
        perm[a], perm[b] = perm[b].item(), perm[a].item()
    return perm


def batch_random_perms(batch_size, n_pos, k_swaps, rng=None):
    """Generate a batch of permutations with k swaps each."""
    perms = torch.stack([random_permutation_with_k_swaps(n_pos, k_swaps, rng)
                         for _ in range(batch_size)])
    return perms  # (B, n_pos) — perm[i] = where position i maps to


def encode_permutation(perms, pos_cos, pos_sin):
    """Encode batch of permutations as VSA hypervectors.

    perm_hv = Σ_i bind(pos_source_i, pos_dest_perm[i])
    """
    B, N = perms.shape
    d = pos_cos.shape[1]

    # Source position phasors: (N, d) → broadcast to (B, N, d)
    src_c = pos_cos.unsqueeze(0).expand(B, -1, -1)
    src_s = pos_sin.unsqueeze(0).expand(B, -1, -1)

    # Destination position phasors: index by perm
    dst_c = pos_cos[perms]  # (B, N, d)
    dst_s = pos_sin[perms]

    # Bind: (src_c + i*src_s)(dst_c + i*dst_s)
    bound_c = src_c * dst_c - src_s * dst_s
    bound_s = src_c * dst_s + src_s * dst_c

    # Superpose
    sr, si = bound_c.sum(1), bound_s.sum(1)  # (B, d)
    mag = (sr ** 2 + si ** 2).sqrt().clamp(min=1e-8)
    return sr / mag, si / mag


def decode_permutation(sr, si, pos_cos, pos_sin):
    """Decode: unbind each source position, classify among all positions.

    Returns logits (B, N_POS, N_POS) — for each source position,
    similarity to each candidate destination position.
    """
    B = sr.shape[0]
    d = pos_cos.shape[1]

    # Unbind source positions: multiply by pos_source.conj()
    ur = sr.unsqueeze(1) * pos_cos.unsqueeze(0) + si.unsqueeze(1) * pos_sin.unsqueeze(0)
    ui = si.unsqueeze(1) * pos_cos.unsqueeze(0) - sr.unsqueeze(1) * pos_sin.unsqueeze(0)
    # ur, ui: (B, N_POS, d) — should contain destination phasor + noise

    # Similarity to all destination positions
    # (B*N_POS, d) @ (d, N_POS)
    ur_flat = ur.reshape(B * N_POS, d)
    ui_flat = ui.reshape(B * N_POS, d)

    logits = (ur_flat @ pos_cos.T + ui_flat @ pos_sin.T) / d
    return logits.reshape(B, N_POS, N_POS)


def test_permutation_recovery(d, k_swaps_list, n_test=1000):
    """Test VSA permutation encoding/decoding at various swap counts."""
    pos_cos, pos_sin = make_position_phasors(N_POS, d)

    results = {}
    for k in k_swaps_list:
        if k > N_POS // 2:
            continue

        rng = torch.Generator().manual_seed(42)
        perms = batch_random_perms(n_test, N_POS, k, rng)

        sr, si = encode_permutation(perms, pos_cos, pos_sin)
        logits = decode_permutation(sr, si, pos_cos, pos_sin)

        preds = logits.argmax(-1)  # (n_test, N_POS)
        pos_acc = (preds == perms).float().mean().item()
        exact = (preds == perms).all(1).float().mean().item()

        # Count how many of the SWAPPED positions are recovered correctly
        identity = torch.arange(N_POS).unsqueeze(0).expand(n_test, -1)
        swapped_mask = (perms != identity)
        if swapped_mask.sum() > 0:
            swap_acc = (preds[swapped_mask] == perms[swapped_mask]).float().mean().item()
        else:
            swap_acc = 1.0

        results[k] = {
            "pos_acc": round(pos_acc, 4),
            "exact_match": round(exact, 4),
            "swap_acc": round(swap_acc, 4),
            "n_swapped_avg": round(swapped_mask.float().sum(1).mean().item(), 1),
        }

        logger.info("  k=%2d swaps: pos_acc=%.1f%% exact=%.1f%% swap_acc=%.1f%% (avg %.1f changed)",
                     k, pos_acc * 100, exact * 100, swap_acc * 100,
                     swapped_mask.float().sum(1).mean().item())

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    k_swaps_list = [1, 2, 4, 8, 16, 32]
    all_results = {}

    for d in [256, 512, 1024, 2048, 4096]:
        logger.info("=" * 60)
        bits = torch.log2(torch.tensor(d / N_POS)).item()
        logger.info("D=%d (%.1f bits/pos, need 6.0 for 64-way)", d, bits)
        logger.info("=" * 60)

        t0 = time.time()
        results = test_permutation_recovery(d, k_swaps_list, n_test=2000)
        elapsed = time.time() - t0

        all_results[f"D={d}"] = {
            "d": d, "bits_per_pos": round(bits, 1),
            "elapsed_s": round(elapsed, 1),
            "by_swaps": results,
        }
        logger.info("D=%d done (%.1fs)\n", d, elapsed)

    # Summary table
    logger.info("=" * 60)
    logger.info("PERMUTATION RECOVERY SUMMARY")
    logger.info("%-6s " + " ".join("k=%-5d" % k for k in k_swaps_list), "D")
    for dkey, dr in all_results.items():
        accs = []
        for k in k_swaps_list:
            if k in dr["by_swaps"]:
                accs.append("%.1f%%" % (dr["by_swaps"][k]["pos_acc"] * 100))
            else:
                accs.append("  -  ")
        logger.info("%-6s " + " ".join("%-7s" % a for a in accs), dkey)

    logger.info("")
    logger.info("EXACT MATCH (all 64 positions correct)")
    logger.info("%-6s " + " ".join("k=%-5d" % k for k in k_swaps_list), "D")
    for dkey, dr in all_results.items():
        accs = []
        for k in k_swaps_list:
            if k in dr["by_swaps"]:
                accs.append("%.1f%%" % (dr["by_swaps"][k]["exact_match"] * 100))
            else:
                accs.append("  -  ")
        logger.info("%-6s " + " ".join("%-7s" % a for a in accs), dkey)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp3_permutation_vsa.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved: %s", out_dir / "exp3_permutation_vsa.json")


if __name__ == "__main__":
    main()
