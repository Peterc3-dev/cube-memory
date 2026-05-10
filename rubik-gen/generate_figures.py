#!/usr/bin/env python3
"""Generate all paper figures from experiment results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def fig1_architecture_comparison():
    """Bar chart: Var% vs architecture for Case Study 1."""
    archs = [
        ("Zero", 0.0, 0),
        ("VSA V1\n(frozen)", 5.0, 35),
        ("VSA V2\n(learned)", 4.8, 35),
        ("Rank-16\nlinear", 5.9, 0.164),
        ("VSA-MoE\n16×128", 14.2, 24),
        ("Learned-MoE\n8×256", 16.2, 21),
        ("Rank-2048\nlinear", 36.6, 21),
        ("Rank-2048\n+ MLP", 38.4, 26),
        ("Full-rank\nceiling", 41.1, 52),
    ]

    names = [a[0] for a in archs]
    vals = [a[1] for a in archs]
    colors = ["#cccccc", "#e74c3c", "#e74c3c", "#3498db",
              "#e67e22", "#f39c12", "#3498db", "#2ecc71", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Variance Captured (%)")
    ax.set_title("Case Study 1: FFN Replacement — Architecture Comparison (Layer 27)")
    ax.axhline(y=4.8, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(8.5, 5.5, "VSA ceiling", color="#e74c3c", fontsize=8, ha="right")
    ax.set_ylim(0, 45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig1_architecture_comparison.png")
    plt.close()
    print("  fig1 done")


def fig3_capacity_wall():
    """Accuracy vs D for exps 1, 1c, 2 showing the capacity wall."""
    with open(RESULTS / "exp1_vsa_capacity.json") as f:
        exp1 = json.load(f)
    with open(RESULTS / "exp1c_factored.json") as f:
        exp1c = json.load(f)

    ds = [512, 1024, 2048, 4096]

    exp1_acc = [exp1[f"D={d}"]["acc"] * 100 for d in ds]
    exp1c_joint = [exp1c[f"D={d}"]["joint_acc"] * 100 for d in ds]
    exp2_acc = [0.6, 0.6, 0.6, 0.61]  # from results

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ds, exp1_acc, "o-", color="#e74c3c", label="Exp 1: Pure VSA (8192-way)", markersize=8)
    ax.plot(ds, exp1c_joint, "s-", color="#e67e22", label="Exp 1c: Factored (128×64)", markersize=8)
    ax.plot(ds, exp2_acc, "^-", color="#9b59b6", label="Exp 2: Real tokens D=4096", markersize=8)
    ax.axhline(y=1/8192*100, color="gray", linestyle=":", label=f"Random chance ({1/8192*100:.3f}%)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("VSA Dimension (D)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Case Study 2: Token Binding — Capacity Wall")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_ylim(-0.1, 2.0)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("Theory: need 13 bits,\nhave ≤6 bits at D=4096",
                xy=(4096, 0.01), xytext=(2048, 1.4),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"))

    fig.tight_layout()
    fig.savefig(FIGS / "fig3_capacity_wall.png")
    plt.close()
    print("  fig3 done")


def fig4_permutation_heatmap():
    """Heatmap: position accuracy (D × k_swaps) from exp 3."""
    with open(RESULTS / "exp3_permutation_vsa.json") as f:
        exp3 = json.load(f)

    ds = [256, 512, 1024, 2048, 4096]
    ks = [1, 2, 4, 8, 16, 32]

    grid = np.zeros((len(ds), len(ks)))
    for i, d in enumerate(ds):
        by_swaps = exp3[f"D={d}"]["by_swaps"]
        for j, k in enumerate(ks):
            grid[i, j] = by_swaps[str(k)]["exact_match"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([str(d) for d in ds])
    ax.set_xlabel("Number of swaps (k)")
    ax.set_ylabel("VSA Dimension (D)")
    ax.set_title("Experiment 3: Permutation Recovery — Exact Match %")

    for i in range(len(ds)):
        for j in range(len(ks)):
            val = grid[i, j]
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if val == 100 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Exact Match %")

    fig.tight_layout()
    fig.savefig(FIGS / "fig4_permutation_heatmap.png")
    plt.close()
    print("  fig4 done")


def fig5_overlap_histogram():
    """Distribution of match rates from exp 4."""
    with open(RESULTS / "exp4_permutation_reconstruction.json") as f:
        exp4 = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4))

    for K, color, marker in [(10, "#e74c3c", "o"), (50, "#3498db", "s"), (200, "#2ecc71", "^")]:
        r = exp4[f"K={K}"]["reconstruction"]
        categories = ["Perfect\n(64/64)", "≥90%\n(≥58/64)", "≥80%\n(≥51/64)", "≥50%\n(≥32/64)"]
        values = [r["perfect_match_pct"]*100, r["above_90_pct"]*100,
                  r["above_80_pct"]*100, r["above_50_pct"]*100]
        x = np.arange(len(categories))
        offset = {"10": -0.25, "50": 0, "200": 0.25}[str(K)]
        ax.bar(x + offset, values, 0.22, label=f"K={K} (mean={r['overall_match_rate']*100:.1f}%)",
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("% of images")
    ax.set_title("Experiment 4: Permutation Reconstruction — Images Are Not Permutations")
    ax.legend()
    ax.set_ylim(0, 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.95, 0.85, "Even at K=200 clusters,\n<4% of images achieve\n≥50% token match",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_overlap_failure.png")
    plt.close()
    print("  fig5 done")


def fig6_signal_decomposition():
    """Stacked bar: what captures the FFN signal."""
    labels = ["Rank-2048\nLinear", "MLP-512\nResidual", "VSA\nContribution", "Uncaptured"]
    values = [36.6, 1.8, 0.4, 61.2]  # linear + MLP-over-linear + VSA-MoE-over-MoE + rest
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#ecf0f1"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = 0
    for label, val, color in zip(labels, values, colors):
        ax.barh(0, val, left=bottom, color=color, edgecolor="black", linewidth=0.5,
                label=f"{label}: {val}%")
        if val > 3:
            ax.text(bottom + val/2, 0, f"{val}%", ha="center", va="center", fontsize=10)
        bottom += val

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Variance Captured (%)")
    ax.set_title("FFN Signal Decomposition — Where Does the Information Live?")
    ax.legend(loc="upper right", bbox_to_anchor=(1, -0.15), ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_signal_decomposition.png")
    plt.close()
    print("  fig6 done")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_architecture_comparison()
    fig3_capacity_wall()
    fig4_permutation_heatmap()
    fig5_overlap_histogram()
    fig6_signal_decomposition()
    print(f"\nAll figures saved to {FIGS}/")
