Learnable kernel intercept and balanced BCE
==========================================

This fork adds an opt-in sampled BCE objective to non-parametric UMAP::

    from umap import UMAP

    model = UMAP(
        objective="bce",
        compatibility_layout=True,
        learnable_bias=True,
        negative_reduction="mean",
        negative_sample_rate=5,
        random_state=42,
    )
    embedding = model.fit_transform(X)
    print(model.kernel_bias_)
    new_embedding = model.transform(X_new)

The default ``objective="umap"`` preserves the upstream implementation. The
new bias and reduction settings do not affect that objective's optimization. The BCE path uses the
compatibility graph and initialization pipeline, followed by its own epoch-batched
Adam optimizer. It currently requires Euclidean output. Input distances may use
any of the ordinary UMAP input metrics. It is intentionally experimental, not a
claim that upstream UMAP is incorrect or that this improves every embedding.

Objective and kernel
--------------------

Write the existing kernel shape parameters as :math:`a,c`, where :math:`c` is
UMAP's public ``b`` parameter. The existing fixed intercept is :math:`\log a`; the trainable offset is
:math:`\beta`, named
``kernel_bias`` initially and ``kernel_bias_`` after fitting:

.. math::

    s(u,v) = -\log a - c\log \max(\|u-v\|^2,10^{-6}),
    \qquad D_\beta(u,v)=\sigma(s(u,v)-\beta).

Away from the numerical distance floor, zero bias gives precisely the existing
UMAP probability :math:`1/(1+a\|u-v\|^{2c})`. The floor makes the loss finite at
coincident points; the implemented derivative of the clamped squared distance
is zero below the floor. No additional repulsive denominator stabilizer is used.

For a sampled positive edge :math:`(i,j)` and :math:`K` negative endpoints:

.. math::

    L_{ij}=-\log\sigma(s_{ij}-\beta)
       -\frac{\gamma}{K}\sum_{k=1}^{K}\log\sigma(-s_{ik}+\beta).

Here :math:`\gamma` is ``repulsion_strength``. ``negative_reduction="sum"``
removes the :math:`1/K`. The loss is then averaged over active positive edges.
With :math:`\gamma=1`, mean reduction implements two equally weighted
expectations; it is twice BCE averaged over two balanced classes. Increasing
:math:`K` changes Monte Carlo precision, not the nominal class weight. It can
still change the optimization trajectory and finite-sample outcome.

Positive edges follow UMAP's existing weighted edge activation schedule.
Negative endpoints are sampled uniformly, with replacement, conditional on the
positive edge's source. During fitting, self endpoints are redrawn so there
are exactly :math:`K` non-self negatives, including possible graph neighbors.
Thus the negative pair distribution has the graph's source marginal; it is
not uniform over all ordered pairs. During transform the source and reference
index spaces differ and no index-based self exclusion is applied.

Relation to the canonical JS bound
---------------------------------

The canonical JS variational lower bound is

.. math::

    4\,\mathrm{JS}(P,Q) \ge
    4\log 2+2\mathbb{E}_P\log\sigma(d)
              +2\mathbb{E}_Q\log\sigma(-d).

With :math:`d=s-\beta`, maximizing this bound is minimizing the balanced
loss above, up to scale and an additive constant. This is a discriminator
objective over graph-positive and noise-pair distributions, not a GAN generator
objective. Equality requires the unrestricted optimal critic
:math:`d^*=\log(P/Q)`; the distance kernel is a restricted critic.

Summing :math:`K` negatives instead changes the unrestricted optimum to
:math:`d^*=\log(P/Q)-\log K`. A scalar bias can express that intercept shift,
but changing class weights also changes the fitted restricted critic and
embedding gradients. Learning a bias alone does not make sum and mean losses
equivalent.

There is also a scale ambiguity: :math:`D_\beta=1/(1+a e^\beta r^{2c})`.
Scaling every embedding coordinate by :math:`t` and replacing the bias by
:math:`\beta-2c\log t` preserves the logits (away from the floor). The bias
therefore does not add independent capacity to unconstrained Euclidean
embeddings; its value and the embedding's global scale are not separately
identifiable. It can change optimization and the resulting calibration.

Optimization and supported operations
------------------------------------

All active positive and negative terms in an epoch see the same coordinates
and bias. Gradients are accumulated for **both endpoints** during fitting,
normalized by the positive count, and passed to Adam. This avoids races on a
shared scalar and applies negative reduction before the update. The embedding
learning rate is ``learning_rate``; the bias uses ``bias_learning_rate=0.1``.
Both decay linearly. This is a separate optimizer from upstream's per-edge
updates, clipping, and repulsive stabilizer.

``transform`` freezes the fitted bias and reference embedding and optimizes
only the new points. Cloning, pickling, refitting, epoch snapshots, and density
output are supported. densMAP, non-Euclidean output, modern hard-negative
options, ``update``, ``inverse_transform``, model arithmetic, and subclasses
that replace the embedding optimizer (including ParametricUMAP) are explicitly
unsupported for BCE. The upstream paths are unchanged.

Ablation and tests
-----------------

Set ``learnable_bias=False`` to hold the bias at ``kernel_bias`` (default zero).
Combine it with either reduction for the four controlled BCE ablations::

    UMAP(objective="bce", compatibility_layout=True,
         learnable_bias=False, negative_reduction="sum")
    UMAP(objective="bce", compatibility_layout=True,
         learnable_bias=True, negative_reduction="sum")
    UMAP(objective="bce", compatibility_layout=True,
         learnable_bias=False, negative_reduction="mean")
    UMAP(objective="bce", compatibility_layout=True,
         learnable_bias=True, negative_reduction="mean")

Run ``python benchmarks/benchmark_balanced_bce.py`` for a reproducible
Iris/Digits comparison over three seeds and negative counts 1, 5, 15, 50,
100, and 500. Use ``--negative-counts`` and ``--seeds`` for smaller subsets.
It includes upstream UMAP as a fifth reference, sharing graph settings, initial
coordinates and epochs. Comparisons among the four BCE variants isolate bias
and reduction; comparisons with upstream additionally change the optimizer and
numerical gradient treatment. Equal epochs are not equal runtime.

The benchmark reports neighborhood trustworthiness, label silhouette, balanced
BCE on the training graph with independently sampled evaluation negatives,
RMS radius, learned bias and runtime. It uses no external downloads. These
small-dataset metrics are exploratory, not a held-out generalization claim.
See ``benchmarks/results/balanced_bce.csv`` and ``balanced_bce_summary.md``.

``umap/tests/test_bce.py`` checks finite-difference gradients, the actual epoch
reduction, extreme logits, coincident points, the zero-bias kernel, the canonical
JS identity, the analytic class-prior shift, fit/transform, serialization,
reproducibility, validation and backward compatibility.

Embedding plots
---------------

The benchmark can save coordinates with ``--embeddings``. The committed seed-0
archive and plots cover all five variants at all six negative counts. Run
``python benchmarks/plot_bce_embeddings.py`` to recreate the panels. Colors show
labels unused during fitting; independently zoomed panels retain equal x/y
scales and report the original RMS radius. See the embedding plots in
``benchmarks/results/balanced_bce_summary.md``.
