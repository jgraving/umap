"""Reproducible ablation; run from the repository root.

python benchmarks/benchmark_balanced_bce.py --output benchmarks/results/balanced_bce.csv
The graph, initialization, epochs and seed are shared across each five-way comparison.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def balanced_loss(model, seed=1729):
    """Weighted graph-positive mean plus an independent uniform-negative mean.

    Evaluation positives are the training graph (not a held-out validation set).
    Negative sources follow the graph's weighted source marginal.
    """
    graph = model.graph_.tocoo()
    weights = graph.data.astype(float)
    weights /= weights.sum()
    z = model.embedding_
    bias = getattr(model, "kernel_bias_", 0.0)

    def logits(i, j):
        r2 = np.maximum(np.sum((z[i] - z[j]) ** 2, axis=1), 1e-6)
        return -np.log(model._a) - model._b * np.log(r2) - bias

    positive = np.sum(weights * np.logaddexp(0, -logits(graph.row, graph.col)))
    rng = np.random.RandomState(seed)
    sources = graph.row[rng.choice(len(weights), 50000, p=weights)]
    targets = rng.randint(z.shape[0] - 1, size=len(sources))
    targets += targets >= sources
    negative = np.logaddexp(0, logits(sources, targets)).mean()
    return positive + negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/balanced_bce.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--negative-counts", type=int, nargs="+", default=[1, 5, 15, 50, 100, 500]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if min(args.negative_counts) < 1:
        parser.error("negative counts must be positive")
    datasets = [
        ("iris", *load_iris(return_X_y=True)),
        ("digits600", *load_digits(return_X_y=True)),
    ]
    modes = [
        ("upstream", "umap", False, "sum"),
        ("fixed_sum", "bce", False, "sum"),
        ("learned_sum", "bce", True, "sum"),
        ("fixed_mean", "bce", False, "mean"),
        ("learned_mean", "bce", True, "mean"),
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = None
        for dataset, x, labels in datasets:
            if len(x) > 600:
                selection = np.random.RandomState(2026).choice(
                    len(x), 600, replace=False
                )
                x, labels = x[selection], labels[selection]
            x = StandardScaler().fit_transform(x).astype(np.float32)
            for seed in args.seeds:
                initialization = (
                    np.random.RandomState(seed)
                    .normal(size=(len(x), 2))
                    .astype(np.float32)
                )
                for k in args.negative_counts:
                    for mode, objective, learn, reduction in modes:
                        model = UMAP(
                            objective=objective,
                            learnable_bias=learn,
                            negative_reduction=reduction,
                            n_epochs=args.epochs,
                            negative_sample_rate=k,
                            random_state=seed,
                            n_jobs=1,
                            compatibility_layout=True,
                            init=initialization.copy(),
                        )
                        started = time.perf_counter()
                        z = model.fit_transform(x)
                        seconds = time.perf_counter() - started
                        row = dict(
                            dataset=dataset,
                            seed=seed,
                            negatives=k,
                            mode=mode,
                            epochs=args.epochs,
                            bias=getattr(model, "kernel_bias_", 0.0),
                            balanced_bce=balanced_loss(model),
                            trustworthiness=trustworthiness(x, z, n_neighbors=10),
                            silhouette=silhouette_score(z, labels),
                            rms_radius=np.sqrt(
                                np.mean(np.sum((z - z.mean(axis=0)) ** 2, axis=1))
                            ),
                            seconds=seconds,
                        )
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=row)
                            writer.writeheader()
                        writer.writerow(row)
                        stream.flush()
                        print(
                            dataset,
                            seed,
                            k,
                            mode,
                            round(row["trustworthiness"], 4),
                            round(row["balanced_bce"], 4),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
