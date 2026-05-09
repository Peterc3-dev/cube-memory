#!/usr/bin/env python3
"""Rubik Gen — Experiment 4: Permutation Reconstruction Quality.

Can real images be well-approximated as permutations of a reference?

For each image:
1. Find nearest cluster reference (by token histogram similarity)
2. Solve optimal assignment via Hungarian algorithm
3. Measure token match rate and decoded image quality

This tests the fundamental viability of the permutation-first architecture.
If token mismatch is high, we need residuals. If it's low, pure permutation works.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

N_POS = 64
N_CODES = 8192


def load_cached_tokens():
    cache = Path(__file__).resolve().parent / "cached_tokens" / "imagenette_tokens.pt"
    if not cache.exists():
        logger.error("No cached tokens at %s — run exp2 first", cache)
        sys.exit(1)
    tokens = torch.load(cache, weights_only=True)
    logger.info("Loaded %d token sequences from %s", len(tokens), cache)
    return tokens


def token_histogram(tokens):
    """Per-image token histogram (bag of tokens)."""
    B = tokens.shape[0]
    hists = torch.zeros(B, N_CODES, dtype=torch.float32)
    for i in range(B):
        hists[i] = torch.bincount(tokens[i], minlength=N_CODES).float()
    return hists


def cluster_by_histogram(tokens, K, seed=42):
    """Simple k-means on token histograms."""
    hists = token_histogram(tokens)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(tokens), generator=rng)[:K]
    centroids = hists[idx].clone()

    for iteration in range(30):
        dists = torch.cdist(hists, centroids)
        assignments = dists.argmin(1)

        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = hists[mask].mean(0)
            else:
                new_centroids[k] = centroids[k]

        shift = (new_centroids - centroids).norm(dim=1).max().item()
        centroids = new_centroids
        if shift < 1e-4:
            logger.info("  k-means converged at iteration %d", iteration)
            break

    counts = torch.bincount(assignments, minlength=K)
    logger.info("  Cluster sizes: min=%d max=%d mean=%d",
                counts.min().item(), counts.max().item(), counts.float().mean().item())
    return assignments, centroids


def pick_reference_for_cluster(tokens, assignments, k):
    """Pick the medoid (most representative member) of cluster k."""
    mask = (assignments == k).nonzero(as_tuple=True)[0]
    if len(mask) == 0:
        return None
    cluster_tokens = tokens[mask]
    hists = token_histogram(cluster_tokens)
    mean_hist = hists.mean(0)
    dists = (hists - mean_hist).norm(dim=1)
    medoid_idx = dists.argmin()
    return cluster_tokens[medoid_idx]


def solve_assignment(target, reference):
    """Find permutation σ minimizing token mismatch via Hungarian algorithm.

    Cost matrix: C[i,j] = 0 if target[i] == reference[j], else 1.
    This maximizes the number of exact token matches.
    """
    cost = (target.unsqueeze(1) != reference.unsqueeze(0)).float()  # (64, 64)
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    perm = torch.zeros(N_POS, dtype=torch.long)
    perm[row_ind] = torch.tensor(col_ind, dtype=torch.long)
    n_matched = N_POS - int(cost[row_ind, col_ind].sum().item())
    return perm, n_matched


def evaluate_reconstruction(tokens, assignments, references, K):
    """For each image, find best permutation of its reference and measure quality."""
    n = len(tokens)
    match_rates = []
    per_cluster_matches = {k: [] for k in range(K)}

    t0 = time.time()
    for i in range(n):
        k = assignments[i].item()
        ref = references[k]
        if ref is None:
            continue

        perm, n_matched = solve_assignment(tokens[i], ref)
        match_rate = n_matched / N_POS
        match_rates.append(match_rate)
        per_cluster_matches[k].append(match_rate)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d (%.1fs) — avg match: %.1f%%",
                        i + 1, n, elapsed, sum(match_rates) / len(match_rates) * 100)

    overall = sum(match_rates) / len(match_rates)
    perfect = sum(1 for r in match_rates if r == 1.0) / len(match_rates)
    above90 = sum(1 for r in match_rates if r >= 0.9) / len(match_rates)
    above80 = sum(1 for r in match_rates if r >= 0.8) / len(match_rates)
    above50 = sum(1 for r in match_rates if r >= 0.5) / len(match_rates)

    cluster_summaries = {}
    for k in range(K):
        rates = per_cluster_matches[k]
        if rates:
            cluster_summaries[k] = {
                "n": len(rates),
                "mean_match": round(sum(rates) / len(rates), 4),
                "perfect": round(sum(1 for r in rates if r == 1.0) / len(rates), 4),
            }

    return {
        "overall_match_rate": round(overall, 4),
        "perfect_match_pct": round(perfect, 4),
        "above_90_pct": round(above90, 4),
        "above_80_pct": round(above80, 4),
        "above_50_pct": round(above50, 4),
        "match_distribution": {
            "mean": round(overall, 4),
            "min": round(min(match_rates), 4),
            "max": round(max(match_rates), 4),
        },
        "per_cluster": cluster_summaries,
    }


def multiset_overlap_baseline(tokens):
    """What's the average token multiset overlap between random pairs?"""
    rng = torch.Generator().manual_seed(42)
    n_pairs = min(2000, len(tokens))
    idx = torch.randperm(len(tokens), generator=rng)[:n_pairs * 2].reshape(n_pairs, 2)

    overlaps = []
    for a, b in idx:
        ca = Counter(tokens[a].tolist())
        cb = Counter(tokens[b].tolist())
        overlap = sum((ca & cb).values())
        overlaps.append(overlap / N_POS)

    mean_overlap = sum(overlaps) / len(overlaps)
    logger.info("Random pair multiset overlap: %.1f%% (of 64 tokens)", mean_overlap * 100)
    return round(mean_overlap, 4)


def within_cluster_overlap(tokens, assignments, K):
    """Average multiset overlap between pairs within each cluster."""
    rng = torch.Generator().manual_seed(42)
    overlaps = []

    for k in range(K):
        mask = (assignments == k).nonzero(as_tuple=True)[0]
        if len(mask) < 2:
            continue
        n_sample = min(200 * 2, len(mask))
        if n_sample < 2:
            continue
        n_sample = n_sample - (n_sample % 2)
        perm = mask[torch.randperm(len(mask), generator=rng)[:n_sample]].reshape(-1, 2)
        for a, b in perm:
            if a == b:
                continue
            ca = Counter(tokens[a].tolist())
            cb = Counter(tokens[b].tolist())
            overlap = sum((ca & cb).values())
            overlaps.append(overlap / N_POS)

    mean_overlap = sum(overlaps) / len(overlaps)
    logger.info("Within-cluster multiset overlap: %.1f%%", mean_overlap * 100)
    return round(mean_overlap, 4)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokens = load_cached_tokens()

    logger.info("=" * 60)
    logger.info("MULTISET OVERLAP BASELINES")
    logger.info("=" * 60)
    random_overlap = multiset_overlap_baseline(tokens)

    all_results = {"random_pair_overlap": random_overlap}

    for K in [10, 50, 200]:
        logger.info("")
        logger.info("=" * 60)
        logger.info("K=%d CLUSTERS", K)
        logger.info("=" * 60)

        logger.info("Clustering by token histogram...")
        assignments, centroids = cluster_by_histogram(tokens, K)

        cluster_overlap = within_cluster_overlap(tokens, assignments, K)

        logger.info("Picking reference (medoid) for each cluster...")
        references = {}
        for k in range(K):
            references[k] = pick_reference_for_cluster(tokens, assignments, k)

        logger.info("Evaluating permutation reconstruction...")
        recon = evaluate_reconstruction(tokens, assignments, references, K)

        logger.info("")
        logger.info("K=%d RESULTS:", K)
        logger.info("  Within-cluster overlap: %.1f%%", cluster_overlap * 100)
        logger.info("  Mean token match after optimal permutation: %.1f%%", recon["overall_match_rate"] * 100)
        logger.info("  Perfect reconstruction (64/64): %.1f%%", recon["perfect_match_pct"] * 100)
        logger.info("  >=90%% match (>=58/64): %.1f%%", recon["above_90_pct"] * 100)
        logger.info("  >=80%% match (>=51/64): %.1f%%", recon["above_80_pct"] * 100)
        logger.info("  >=50%% match (>=32/64): %.1f%%", recon["above_50_pct"] * 100)

        all_results[f"K={K}"] = {
            "K": K,
            "within_cluster_overlap": cluster_overlap,
            "reconstruction": recon,
        }

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("%-6s %-12s %-12s %-10s %-10s %-10s",
                "K", "Overlap%", "MatchRate%", "Perfect%", ">=90%", ">=80%")
    for K in [10, 50, 200]:
        r = all_results[f"K={K}"]
        rc = r["reconstruction"]
        logger.info("%-6d %-12.1f %-12.1f %-10.1f %-10.1f %-10.1f",
                     K, r["within_cluster_overlap"] * 100,
                     rc["overall_match_rate"] * 100,
                     rc["perfect_match_pct"] * 100,
                     rc["above_90_pct"] * 100,
                     rc["above_80_pct"] * 100)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp4_permutation_reconstruction.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved: %s", out_dir / "exp4_permutation_reconstruction.json")


if __name__ == "__main__":
    main()
