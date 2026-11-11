"""Experimental sampled BCE layout with a calibrated UMAP kernel.

The shape parameter ``b`` is distinct from the learned scalar ``kernel_bias``.
See doc/balanced_bce.rst for the objective and sampling distribution.
"""

import numba
import numpy as np
from tqdm.auto import tqdm

from umap.utils import tau_rand_int


@numba.njit(cache=True)
def bce_pair(dist_squared, a, b, kernel_bias, positive):
    """Return stable loss, coordinate-gradient coefficient and bias gradient.

    Multiply the coefficient by (head - tail) for the head gradient. A squared
    distance floor regularizes the logit at coincident points; its derivative
    is zero below the floor, consistently with this loss.
    """
    radius = max(dist_squared, 1e-6)
    logit = -np.log(a) - b * np.log(radius) - kernel_bias
    if logit >= 0.0:
        exp_neg = np.exp(-logit)
        probability = 1.0 / (1.0 + exp_neg)
        complement = exp_neg / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(logit)
        probability = exp_pos / (1.0 + exp_pos)
        complement = 1.0 / (1.0 + exp_pos)
    if positive:
        loss = max(-logit, 0.0) + np.log1p(np.exp(-abs(logit)))
        bias_gradient = complement
    else:
        loss = max(logit, 0.0) + np.log1p(np.exp(-abs(logit)))
        bias_gradient = -probability
    coefficient = 2.0 * b * bias_gradient / radius
    if dist_squared < 1e-6:
        coefficient = 0.0
    return loss, coefficient, bias_gradient


@numba.njit(cache=True)
def _bce_epoch_gradients(
    embedding,
    reference,
    head,
    tail,
    epochs_per_sample,
    next_sample,
    epoch,
    a,
    b,
    kernel_bias,
    negative_sample_rate,
    mean_negatives,
    repulsion_strength,
    rng_state,
    move_other,
):
    # All terms see the same coordinates and bias. In particular, reduction
    # happens BEFORE the optimizer update, not after K sequential SGD steps.
    gradient = np.zeros_like(embedding)
    bias_gradient = 0.0
    loss = 0.0
    count = 0
    negative_weight = repulsion_strength
    if mean_negatives:
        negative_weight /= negative_sample_rate
    for edge in range(head.size):
        if next_sample[edge] > epoch:
            continue
        i, j = head[edge], tail[edge]
        count += 1
        for sample in range(negative_sample_rate + 1):
            positive = sample == 0
            if positive:
                k = j
                weight = 1.0
            else:
                k = tau_rand_int(rng_state) % reference.shape[0]
                # Fit draws exactly K non-self negatives, with replacement.
                # During transform the two index spaces are unrelated.
                while move_other and k == i:
                    k = tau_rand_int(rng_state) % reference.shape[0]
                weight = negative_weight
            delta = embedding[i] - reference[k]
            pair_loss, coefficient, pair_bias = bce_pair(
                np.dot(delta, delta), a, b, kernel_bias, positive
            )
            loss += weight * pair_loss
            bias_gradient += weight * pair_bias
            for d in range(embedding.shape[1]):
                value = weight * coefficient * delta[d]
                gradient[i, d] += value
                if move_other:
                    gradient[k, d] -= value
        next_sample[edge] += epochs_per_sample[edge]
    if count:
        gradient /= count
        bias_gradient /= count
        loss /= count
    return gradient, bias_gradient, loss, count


def optimize_layout_bce(
    embedding,
    reference,
    head,
    tail,
    n_epochs,
    epochs_per_sample,
    a,
    b,
    rng_state,
    negative_sample_rate=5,
    initial_alpha=1.0,
    repulsion_strength=1.0,
    kernel_bias=0.0,
    learnable_bias=True,
    negative_reduction="mean",
    bias_learning_rate=0.1,
    move_other=True,
    verbose=False,
):
    """Optimize sampled BCE using epoch-batched Adam and a scalar bias.

    Returns (embedding, bias). Reference points and bias remain fixed during
    transform (``move_other=False, learnable_bias=False``).
    """
    next_sample = epochs_per_sample.copy()
    first = np.zeros_like(embedding)
    second = np.zeros_like(embedding)
    bias_first = bias_second = 0.0
    step = 0
    requested_epochs = n_epochs if isinstance(n_epochs, list) else None
    total_epochs = max(n_epochs) if requested_epochs is not None else n_epochs
    snapshots = []
    if requested_epochs is not None and 0 in requested_epochs:
        snapshots.append(embedding.copy())
    for epoch in tqdm(range(total_epochs), disable=not verbose, desc="BCE layout"):
        gradient, bias_gradient, _, count = _bce_epoch_gradients(
            embedding,
            reference,
            head,
            tail,
            epochs_per_sample,
            next_sample,
            epoch,
            a,
            b,
            kernel_bias,
            negative_sample_rate,
            negative_reduction == "mean",
            repulsion_strength,
            rng_state,
            move_other,
        )
        if count:
            step += 1
            decay = 1.0 - epoch / max(total_epochs, 1)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            embedding -= (
                initial_alpha
                * decay
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
            if learnable_bias:
                bias_first = 0.9 * bias_first + 0.1 * bias_gradient
                bias_second = 0.999 * bias_second + 0.001 * bias_gradient**2
                kernel_bias -= (
                    bias_learning_rate
                    * decay
                    * (bias_first / (1 - 0.9**step))
                    / (np.sqrt(bias_second / (1 - 0.999**step)) + 1e-8)
                )
        if requested_epochs is not None and epoch + 1 in requested_epochs:
            snapshots.append(embedding.copy())
    if requested_epochs is not None:
        return snapshots or [embedding.copy()], float(kernel_bias)
    return embedding, float(kernel_bias)
