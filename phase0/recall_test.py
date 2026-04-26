"""Synthetic associative-recall test for the FHRR algebra.

Goal: at the operating point (d=1024, m=256, p=3), verify that
`unbind(bind(role_keys), one_key) ≈ remaining_bind` and that
cleanup recovers the correct role index above the noise floor.

This is the Phase 0 validation gate. If recall here is below 95%
at our target operating point, the architecture is broken and
nothing in Phase 1+ will save it.

No model training. Pure algebra check.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from fhrr import bind, cleanup, random_codebook, superpose, unbind


@dataclass
class Result:
    config: str
    n_trials: int
    direct_recall: float          # bind(a,b) — unbind by a — recover b
    triple_recall: float          # bind(a,b,c) — unbind by a — recover bind(b,c)
    superpose_recall: float       # mean recall over a bundled set
    superpose_size: int


def role_axis_recall(d: int, m: int, p: int, n_trials: int,
                     superpose_size: int, device: str, seed: int) -> Result:
    """Run the full sweep at one operating point."""
    g = torch.Generator(device=device).manual_seed(seed)

    # p independent role-axis codebooks, each with m vectors of dim d.
    axes = [random_codebook(m, d, device=device, seed=seed + ax) for ax in range(p)]

    # Trial 1 — direct bind/unbind: bind(a, b), unbind by a, recover b.
    direct_hits = 0
    for _ in range(n_trials):
        i_a = int(torch.randint(0, m, (1,), generator=g, device=device))
        i_b = int(torch.randint(0, m, (1,), generator=g, device=device))
        a = axes[0][i_a]
        b = axes[1][i_b]
        bound = bind(a, b)
        recovered = unbind(bound, a)
        idx, _ = cleanup(recovered.unsqueeze(0), axes[1])
        if int(idx) == i_b:
            direct_hits += 1

    # Trial 2 — depth-3 bind, unbind one role, recover the *bind of the
    # remaining two*. Cleanup against a synthetic codebook of all
    # m^2 b-c pairs would be huge; instead, verify each remaining role
    # individually (unbind twice) and check both indices match.
    triple_hits = 0
    for _ in range(n_trials):
        i_a = int(torch.randint(0, m, (1,), generator=g, device=device))
        i_b = int(torch.randint(0, m, (1,), generator=g, device=device))
        i_c = int(torch.randint(0, m, (1,), generator=g, device=device))
        a, b, c = axes[0][i_a], axes[1][i_b], axes[2][i_c]
        bound = bind(a, b, c)
        # Unbind two roles; the remainder should be the third.
        rec_b = unbind(unbind(bound, a), c)
        rec_c = unbind(unbind(bound, a), b)
        idx_b, _ = cleanup(rec_b.unsqueeze(0), axes[1])
        idx_c, _ = cleanup(rec_c.unsqueeze(0), axes[2])
        if int(idx_b) == i_b and int(idx_c) == i_c:
            triple_hits += 1

    # Trial 3 — superposition of `superpose_size` bound pairs,
    # unbind one query key, expect to recover the value with which
    # it was bound. This tests the *content-addressable memory*
    # property (a Memory Layer is essentially this).
    sup_total = 0
    sup_hits = 0
    for _ in range(n_trials):
        # k random key-value pairs from axes[0] and axes[1]
        k = superpose_size
        ki = torch.randint(0, m, (k,), generator=g, device=device)
        vi = torch.randint(0, m, (k,), generator=g, device=device)
        keys = axes[0][ki]
        vals = axes[1][vi]
        bundle = superpose(bind(keys, vals))  # shape (d,)
        for j in range(k):
            recovered = unbind(bundle, axes[0][int(ki[j])])
            idx, _ = cleanup(recovered.unsqueeze(0), axes[1])
            sup_total += 1
            if int(idx) == int(vi[j]):
                sup_hits += 1

    return Result(
        config=f"d={d} m={m} p={p}",
        n_trials=n_trials,
        direct_recall=direct_hits / n_trials,
        triple_recall=triple_hits / n_trials,
        superpose_recall=sup_hits / max(sup_total, 1),
        superpose_size=superpose_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"device: {args.device}, trials each: {args.n_trials}\n")

    # Sweep across dimensions, codebook sizes, and bundle sizes.
    configs = [
        # (d, m, p, superpose_size)
        (256, 64, 3, 16),
        (512, 128, 3, 32),
        (1024, 256, 3, 32),
        (1024, 256, 3, 64),
        (1024, 256, 3, 128),
        (2048, 256, 3, 64),
    ]

    print(f"{'config':<28} {'direct':>8} {'depth-3':>8} {'super':>8} {'super_n':>8}")
    print("-" * 70)
    for d, m, p, k in configs:
        r = role_axis_recall(d, m, p, args.n_trials, k, args.device, args.seed)
        print(f"{r.config:<28} {r.direct_recall:>8.3f} {r.triple_recall:>8.3f} "
              f"{r.superpose_recall:>8.3f} {r.superpose_size:>8}")


if __name__ == "__main__":
    main()
