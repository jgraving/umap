"""Plot a fixed-seed embedding sweep saved by benchmark_balanced_bce.py.

python benchmarks/benchmark_balanced_bce.py --seeds 0 --embeddings benchmarks/results/embeddings_seed0.npz
python benchmarks/plot_bce_embeddings.py

Requires matplotlib. Coordinates are centered and isotropically rescaled for
viewing; their raw RMS radius is displayed so scale changes remain visible.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODES = ["upstream", "fixed_sum", "learned_sum", "fixed_mean", "learned_mean"]
TITLES = [
    "Upstream UMAP",
    "Fixed intercept\nSum negatives",
    "Learned intercept\nSum negatives",
    "Fixed intercept\nMean negatives",
    "Learned intercept\nMean negatives",
]


def plot_grid(archive, datasets, counts, seed, output, focus=False):
    rows = [(dataset, k) for dataset in datasets for k in counts]
    fig, axes = plt.subplots(
        len(rows), 5, figsize=(15, 2.6 * len(rows) + 1.3), squeeze=False
    )
    handles = []
    for row, (dataset, k) in enumerate(rows):
        labels = archive[f"{dataset}__labels"]
        classes = np.unique(labels)
        palette = plt.get_cmap("tab10")
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            z = archive[f"{dataset}__seed{seed}__k{k}__{mode}"]
            centered = z - z.mean(axis=0)
            rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
            displayed = centered / max(rms, 1e-12)
            for label in classes:
                selected = labels == label
                ax.scatter(
                    *displayed[selected].T,
                    s=10 if dataset == "iris" else 5,
                    color=palette(int(label)),
                    alpha=0.85,
                    linewidths=0,
                    rasterized=True,
                )
            # Equal x/y scales within each panel; independent isotropic view
            # scale across panels. No rotation, anisotropic stretch, or clipping.
            limit = max(float(np.max(np.abs(displayed))) * 1.06, 1.0)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dddddd")
            ax.set_xlabel(f"raw RMS radius = {rms:.2f}", fontsize=8, color="#555555")
            if row == 0:
                ax.set_title(TITLES[col], fontsize=11, fontweight="bold", pad=12)
            if col == 0:
                title = "Iris" if dataset == "iris" else "Digits"
                ax.set_ylabel(
                    f"{title}\nK = {k}", fontsize=11, fontweight="bold", labelpad=12
                )
        if dataset == "iris":
            class_names = ["setosa", "versicolor", "virginica"]
        else:
            class_names = [str(label) for label in classes]
        if not focus:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=palette(int(label)),
                    markeredgewidth=0,
                    markersize=6,
                    label=name,
                )
                for label, name in zip(classes, class_names)
            ]
    title = (
        "Embeddings at 500 negatives per positive"
        if focus
        else (
            f"{'Iris' if datasets[0]=='iris' else 'Digits'} embeddings across negative counts"
        )
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.987)
    caption = (
        f"Seed {seed} · {int(archive['__epochs'])} epochs · colors show class labels (unused during fitting)\n"
        "Panels centered and independently zoomed with equal x/y scale; raw RMS radius shown. Orientations are arbitrary."
    )
    fig.text(
        0.5,
        0.018 if focus else 0.013,
        caption,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045 if focus else 0.043),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
        )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.86 if focus else 0.91,
        bottom=0.14 if focus else 0.09,
        hspace=0.28,
        wspace=0.12,
    )
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results/embeddings_seed0.npz")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as archive:
        for dataset in ["iris", "digits600"]:
            plot_grid(
                archive,
                [dataset],
                [1, 5, 15, 50, 100, 500],
                args.seed,
                output / f"embeddings_{dataset}_seed{args.seed}.png",
            )
        plot_grid(
            archive,
            ["iris", "digits600"],
            [500],
            args.seed,
            output / f"embeddings_k500_seed{args.seed}.png",
            focus=True,
        )


if __name__ == "__main__":
    main()
