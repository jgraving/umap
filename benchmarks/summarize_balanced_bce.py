"""Render the ablation CSV as tables and a figure (requires pandas/matplotlib).

Run after benchmark_balanced_bce.py, from the repository root.
"""

import argparse
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy, sklearn, numba

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="benchmarks/results/balanced_bce.csv")
parser.add_argument("--preview-png")
args = parser.parse_args()
root = Path(args.input).parent
x = pd.read_csv(args.input)
counts = sorted(x.negatives.unique())
max_count = max(counts)
modes = ["upstream", "fixed_sum", "learned_sum", "fixed_mean", "learned_mean"]
names = {
    "upstream": "Upstream UMAP",
    "fixed_sum": "Fixed intercept / sum",
    "learned_sum": "Learned intercept / sum",
    "fixed_mean": "Fixed intercept / mean",
    "learned_mean": "Learned intercept / mean",
}
colors = ["#737373", "#ca6b3f", "#a14f85", "#2f69ad", "#15866c"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
for col, dataset in enumerate(["iris", "digits600"]):
    subset = x[x.dataset == dataset]
    for mode, color in zip(modes, colors):
        grouped = subset[subset["mode"] == mode].groupby("negatives")
        for row, metric in enumerate(["trustworthiness", "balanced_bce"]):
            mean = grouped[metric].mean()
            std = grouped[metric].std()
            axes[row, col].errorbar(
                mean.index,
                mean,
                yerr=std,
                color=color,
                marker="o",
                capsize=3,
                label=names[mode],
                linewidth=1.7,
            )
    axes[0, col].set_title(
        "Iris (150 points)" if col == 0 else "Digits (600 points)",
        loc="left",
        fontweight="bold",
    )
    for ax in axes[:, col]:
        ax.set_xscale("log")
        ax.set_xticks(counts, [str(k) for k in counts])
        ax.set_xlabel("Negatives per positive edge (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
axes[0, 0].set_ylabel("Trustworthiness @ 10 (higher is better)")
axes[1, 0].set_ylabel("Balanced BCE (lower is better)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle(
    "Negative averaging helps; learning the intercept has mixed effects",
    fontsize=14,
    fontweight="bold",
)
fig.savefig(root / "balanced_bce.svg")
svg = root / "balanced_bce.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
if args.preview_png:
    fig.savefig(args.preview_png, dpi=130)

lines = [
    "# Learnable kernel intercept and negative reduction: ablation",
    "",
    f"{len(x)} fits: 2 datasets × 3 seeds × {len(counts)} negative counts × 5 variants. This is an exploratory comparison, not evidence of universal improvement.",
    "",
    "![Ablation across negative counts](balanced_bce.svg)",
    "",
    "## Findings",
    "",
    "- Within the common BCE optimizer, mean reduction consistently beats sum reduction on mean trustworthiness for all tested K > 1 in both datasets, with either fixed or learned intercepts. At K = 1, the paired sum/mean results are identical.",
    "- Learning the intercept does not consistently improve neighborhood quality over the fixed-intercept mean objective. At K = 5 it slightly lowers mean trustworthiness on both datasets; its balanced BCE is slightly lower on Iris and higher on Digits.",
    "- The strongest result here is for negative normalization, not extra representational capacity from the intercept. The intercept and global coordinate scale are redundant away from the numerical floor.",
    "- Upstream is an external reference: it also differs in optimizer, clipping, repulsive stabilization and treatment of negative endpoints. Only the four BCE variants isolate the requested switches.",
    "",
    "## K = 5: mean ± sample standard deviation across three seeds",
    "",
    "| Dataset | Variant | Trustworthiness ↑ | Balanced BCE ↓ | Silhouette ↑ | Learned bias |",
    "|---|---|---:|---:|---:|---:|",
]
for dataset in ["iris", "digits600"]:
    for mode in modes:
        rows = x[(x.dataset == dataset) & (x["mode"] == mode) & (x.negatives == 5)]
        cells = [
            f"{rows[m].mean():.4f} ± {rows[m].std():.4f}"
            for m in ["trustworthiness", "balanced_bce", "silhouette", "bias"]
        ]
        lines.append("| " + " | ".join([dataset, names[mode], *cells]) + " |")
lines += [
    "",
    f"## K = {max_count}: mean ± sample standard deviation across three seeds",
    "",
    "| Dataset | Variant | Trustworthiness ↑ | Balanced BCE ↓ | Silhouette ↑ | Learned bias |",
    "|---|---|---:|---:|---:|---:|",
]
for dataset in ["iris", "digits600"]:
    for mode in modes:
        rows = x[
            (x.dataset == dataset) & (x["mode"] == mode) & (x.negatives == max_count)
        ]
        cells = [
            f"{rows[m].mean():.4f} ± {rows[m].std():.4f}"
            for m in ["trustworthiness", "balanced_bce", "silhouette", "bias"]
        ]
        lines.append("| " + " | ".join([dataset, names[mode], *cells]) + " |")
lines += [
    "",
    "## Trustworthiness versus the number of negatives",
    "",
    "| Dataset | Variant | " + " | ".join(f"K = {k}" for k in counts) + " |",
    "|---|---|" + "---:|" * len(counts),
]
for dataset in ["iris", "digits600"]:
    for mode in modes:
        rows = x[(x.dataset == dataset) & (x["mode"] == mode)]
        cells = [
            f"{rows[rows.negatives==k].trustworthiness.mean():.4f}" for k in counts
        ]
        lines.append("| " + " | ".join([dataset, names[mode], *cells]) + " |")
lines += [
    "",
    "## Protocol",
    "",
    "- Upstream base: `lmcinnes/umap@8c925ee2b2a0b8425d4a240dfbfb91eadec43c93` (2026-11-11).",
    "- Iris: all 150 rows. Digits: 600 rows sampled without replacement with seed 2026. Features standardized before fitting; labels are used only to calculate silhouette.",
    "- Seeds: 0, 1, 2. Negative counts: 1, 5, 15, 50, 100, 500. Epochs: 200. Other common settings: n_neighbors=15, min_dist=0.1, repulsion_strength=1, learning_rate=1, n_jobs=1, compatibility_layout=True.",
    "- For each dataset/seed all variants receive the same Gaussian initialization, passed through the same compatibility rescaling. Graph construction and edge schedules use the same settings. The four BCE variants use the same negative draws for a given seed and K.",
    "- Bias initialization: 0. Bias Adam learning rate: 0.1. Standard Adam moments (0.9, 0.999), epsilon 1e-8, linearly decayed learning rates. No hyperparameter search was performed.",
    "- Balanced BCE is evaluated on the full training fuzzy graph, weighted by its edge weights, plus 50,000 independent uniform non-self negatives (seed 1729; source drawn from the weighted graph marginal). It is not held-out positive-edge performance. The training schedule approximates graph weighting and prunes sufficiently small edges.",
    "- Trustworthiness uses 10 neighbors in the standardized input space. Silhouette uses the class labels. The CSV also records RMS radius, fitted intercept and wall time; timings include first-call JIT compilation and are not a rigorous speed benchmark.",
    "- Negatives are sampled with replacement, so K can exceed the number of distinct endpoints. The high-K sweep tests a more precise estimate of the same noise expectation, not 500 distinct negatives per edge.",
    "- Raw results: [balanced_bce.csv](balanced_bce.csv). Reproduce from the repository root with `python benchmarks/benchmark_balanced_bce.py`.",
    "- Regenerate the report and figure with `python benchmarks/summarize_balanced_bce.py` (requires pandas and matplotlib from the plot extra).",
    "- The reported sample standard deviations describe only these three seeds, not statistical significance.",
    "",
    "## Environment",
    "",
    f"Python {platform.python_version()}; NumPy {np.__version__}; SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}; Numba {numba.__version__}.",
    "",
    "## Objective interpretation",
    "",
    "For balanced BCE L (sum of the two class expectations), the canonical JS lower bound is `4 log(2) - 2 L`. See [the objective documentation](../../doc/balanced_bce.rst) for the derivation, fixed-versus-learned intercept distinction, sampling distribution and limitations.",
    "",
]
if (root / "embeddings_seed0.npz").exists():
    lines.append(
        "\n## Embedding plots\n\nThese plots show **seed 0**, selected in advance rather than by visual quality,\nwith the same data, initialization, 200 epochs, and settings as the metric sweep.\nColors denote class labels, which were not used in fitting. Each panel is\ncentered and independently zoomed using equal x/y scale; the raw RMS radius is\nprinted below it. Orientations are arbitrary, and no points are cropped.\n\n### 500 negatives per positive\n\n![Both datasets at K=500](embeddings_k500_seed0.png)\n\n### Iris: all negative counts\n\n![Iris embedding sweep](embeddings_iris_seed0.png)\n\n### Digits: all negative counts\n\n![Digits embedding sweep](embeddings_digits600_seed0.png)\n\nThe saved [coordinates and labels](embeddings_seed0.npz) reproduce these plots:\n\n```bash\npython benchmarks/plot_bce_embeddings.py\n```\n\nTo regenerate the embeddings too:\n\n```bash\npython benchmarks/benchmark_balanced_bce.py --seeds 0 --output /tmp/embedding_metrics.csv --embeddings benchmarks/results/embeddings_seed0.npz\npython benchmarks/plot_bce_embeddings.py\n```\n"
    )
(root / "balanced_bce_summary.md").write_text("\n".join(lines))
