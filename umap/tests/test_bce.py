import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.datasets import load_iris

from umap import UMAP
from umap.bce import bce_pair, _bce_epoch_gradients, optimize_layout_bce


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("bias", [-3.0, 0.0, 4.0])
def test_pair_gradients_finite_difference(positive, bias):
    x = np.array([0.7, -1.2])
    y = np.array([-0.2, 0.4])
    a, b = 1.7, 0.8
    delta = x - y
    loss, coefficient, dbias = bce_pair(delta @ delta, a, b, bias, positive)

    def objective(x, bias):
        z = -np.log(a) - b * np.log(np.sum((x - y) ** 2)) - bias
        return np.logaddexp(0, -z if positive else z)

    assert_allclose(loss, objective(x, bias))
    eps = 1e-5
    numerical = np.array(
        [
            (objective(x + eps * e, bias) - objective(x - eps * e, bias)) / (2 * eps)
            for e in np.eye(2)
        ]
    )
    assert_allclose(coefficient * delta, numerical, rtol=1e-6, atol=1e-9)
    assert_allclose(
        dbias,
        (objective(x, bias + eps) - objective(x, bias - eps)) / (2 * eps),
        rtol=1e-6,
    )


@pytest.mark.parametrize("bias", [-1000.0, 1000.0])
@pytest.mark.parametrize("distance", [0.0, 1e-12, 1.0, 1e20])
def test_pair_stability(bias, distance):
    for positive in [True, False]:
        values = bce_pair(distance, 1.7, 0.8, bias, positive)
        assert np.isfinite(values).all()
        if distance < 1e-6:
            assert values[1] == 0


def gradients(k, mean, bias=0.3, embedding=None):
    if embedding is None:
        embedding = np.array([[0.0, 0.0], [1.0, 2.0]])
    return _bce_epoch_gradients(
        embedding,
        embedding,
        np.array([0]),
        np.array([1]),
        np.ones(1),
        np.zeros(1),
        0,
        1.3,
        0.9,
        bias,
        k,
        mean,
        1.0,
        np.array([12345, 67890, 54321], dtype=np.int64),
        True,
    )


def test_mean_invariant_to_duplicate_negatives_in_actual_epoch():
    # With two vertices every sampled non-self negative is the same point.
    one = gradients(1, True)
    many = gradients(17, True)
    for i in range(3):
        assert_allclose(one[i], many[i], atol=1e-12)
    assert one[3] == many[3] == 1
    assert not np.allclose(gradients(1, False)[0], gradients(17, False)[0])


def test_epoch_coordinates_and_bias_match_finite_difference():
    x = np.array([[0.1, 0.2], [1.2, -0.7]])
    g, db, loss, _ = gradients(5, True, embedding=x)
    eps = 1e-5
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical = (
            gradients(5, True, embedding=plus)[2]
            - gradients(5, True, embedding=minus)[2]
        ) / (2 * eps)
        assert_allclose(g[index], numerical, rtol=1e-6, atol=1e-9)
    numerical = (
        gradients(5, True, bias=0.3 + eps, embedding=x)[2]
        - gradients(5, True, bias=0.3 - eps, embedding=x)[2]
    ) / (2 * eps)
    assert_allclose(db, numerical, rtol=1e-6)
    assert_allclose(g.sum(axis=0), 0, atol=1e-12)


def test_negative_sum_induces_log_k_optimum_shift():
    score, k = 1.7, 9

    def loss(bias, weight):
        return np.logaddexp(0, bias - score) + weight * np.logaddexp(0, score - bias)

    mean = minimize_scalar(lambda bias: loss(bias, 1)).x
    summed = minimize_scalar(lambda bias: loss(bias, k)).x
    assert_allclose(mean, score, atol=1e-6)
    assert_allclose(summed - mean, np.log(k), atol=1e-6)


def test_js_bound_tight_at_log_density_ratio():
    p = np.array([0.1, 0.3, 0.6])
    q = np.array([0.5, 0.4, 0.1])
    d = np.log(p / q)
    balanced_bce = np.sum(p * np.logaddexp(0, -d) + q * np.logaddexp(0, d))
    mixture = (p + q) / 2
    js = 0.5 * np.sum(p * np.log(p / mixture) + q * np.log(q / mixture))
    assert_allclose(4 * np.log(2) - 2 * balanced_bce, 4 * js)
    arbitrary = d + 0.7
    other = np.sum(p * np.logaddexp(0, -arbitrary) + q * np.logaddexp(0, arbitrary))
    assert 4 * np.log(2) - 2 * other <= 4 * js


def test_zero_bias_recovers_original_kernel():
    r2, a, b = 2.4, 1.7, 0.8
    loss, _, _ = bce_pair(r2, a, b, 0.0, True)
    assert_allclose(np.exp(-loss), 1 / (1 + a * r2**b))


@pytest.fixture(scope="module")
def iris():
    return load_iris().data.astype(np.float32)


def model(**kwargs):
    return UMAP(
        objective="bce",
        n_epochs=40,
        random_state=42,
        n_jobs=1,
        compatibility_layout=True,
        **kwargs,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_fit_with_and_without_bias_and_transform(iris, reduction):
    learned = model(negative_reduction=reduction).fit(iris[:120])
    fixed = model(learnable_bias=False, negative_reduction=reduction).fit(iris[:120])
    assert np.isfinite(learned.embedding_).all()
    assert abs(learned.kernel_bias_) > 1e-4
    assert fixed.kernel_bias_ == 0
    assert not np.allclose(learned.embedding_, fixed.embedding_)
    before = learned.embedding_.copy()
    bias_before = learned.kernel_bias_
    transformed = learned.transform(iris[120:])
    assert transformed.shape == (30, 2)
    assert np.isfinite(transformed).all()
    assert_array_equal(before, learned.embedding_)
    assert bias_before == learned.kernel_bias_
    assert_array_equal(
        transformed, pickle.loads(pickle.dumps(learned)).transform(iris[120:])
    )
    repeated = clone(learned).fit(iris[:120])
    assert_array_equal(learned.embedding_, repeated.embedding_)
    assert learned.kernel_bias_ == repeated.kernel_bias_


def test_default_path_is_unchanged(iris):
    common = dict(n_epochs=20, random_state=42, n_jobs=1, compatibility_layout=True)
    default = UMAP(**common).fit_transform(iris)
    explicit = UMAP(
        objective="umap",
        kernel_bias=9,
        learnable_bias=False,
        negative_reduction="sum",
        **common,
    ).fit_transform(iris)
    assert_array_equal(default, explicit)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"negative_sample_rate": 0}, "positive integer"),
        ({"negative_sample_rate": 1.5}, "positive integer"),
        ({"kernel_bias": np.nan}, "finite"),
        ({"kernel_bias": True}, "finite"),
        ({"bias_learning_rate": 0}, "positive"),
        ({"negative_reduction": "other"}, "negative_reduction"),
        ({"learnable_bias": 1}, "boolean"),
        ({"densmap": True}, "Euclidean"),
        ({"output_metric": "manhattan"}, "Euclidean"),
        ({"a": 0}, "kernel a"),
        ({"b": -1}, "kernel b"),
        ({"exclude_graph_neighbors": True}, "uniform"),
    ],
)
def test_invalid_parameters(iris, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model(**kwargs).fit(iris)


def test_modern_optimizer_rejected(iris):
    with pytest.raises(ValueError, match="requires compatibility_layout"):
        UMAP(objective="bce", optimizer="adam", n_epochs=10).fit(iris)


def test_unsupported_operations_fail_explicitly(iris):
    fitted = model().fit(iris)
    with pytest.raises(NotImplementedError):
        fitted.update(iris[:3])
    with pytest.raises(NotImplementedError):
        fitted.inverse_transform(fitted.embedding_[:3])
    for operation in [
        lambda: fitted + fitted,
        lambda: fitted * fitted,
        lambda: fitted - fitted,
    ]:
        with pytest.raises(NotImplementedError):
            operation()


def test_snapshot_epochs(iris):
    fitted = UMAP(
        objective="bce",
        compatibility_layout=True,
        random_state=42,
        n_jobs=1,
        n_epochs=[0, 10, 20],
    ).fit(iris)
    assert len(fitted.embedding_list_) == 3
    assert_array_equal(fitted.embedding_, fitted.embedding_list_[-1])


@pytest.mark.parametrize("reduction,k", [("mean", 9), ("sum", 9)])
def test_learning_bias_calibrates_fixed_embeddings(reduction, k):
    embedding = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    a, shape = 1.3, 0.9
    score = -np.log(a) - shape * np.log(5.0)
    expected = score + (np.log(k) if reduction == "sum" else 0.0)
    _, learned = optimize_layout_bce(
        embedding,
        embedding,
        np.array([0]),
        np.array([1]),
        600,
        np.ones(1),
        a,
        shape,
        np.array([12345, 67890, 54321], dtype=np.int64),
        negative_sample_rate=k,
        initial_alpha=0.0,
        kernel_bias=4.0,
        negative_reduction=reduction,
    )
    assert_allclose(learned, expected, atol=2e-3)
    weight = k if reduction == "sum" else 1

    def objective(bias):
        return np.logaddexp(0, bias - score) + weight * np.logaddexp(0, score - bias)

    assert objective(learned) < objective(4.0)
    assert_array_equal(embedding, [[0.0, 0.0], [1.0, 2.0]])


def test_refit_resets_fitted_bias(iris):
    fitted = model().fit(iris)
    original = fitted.kernel_bias_
    fitted.fit(iris)
    assert fitted.kernel_bias_ == original
    fitted.set_params(objective="umap").fit(iris)
    assert not hasattr(fitted, "kernel_bias_")


def test_single_sample_has_initial_bias():
    fitted = model(kernel_bias=0.7).fit(np.zeros((1, 3)))
    assert fitted.kernel_bias_ == 0.7


def test_overridden_embedding_optimizer_is_rejected(iris):
    class CustomUMAP(UMAP):
        def _fit_embed_data(self, *args, **kwargs):
            raise AssertionError("must not silently ignore BCE")

    with pytest.raises(ValueError, match="subclasses overriding"):
        CustomUMAP(objective="bce", compatibility_layout=True).fit(iris)


@pytest.mark.parametrize("mode", ["sparse", "unique", "density", "cosine"])
def test_bce_fit_input_modes(iris, mode):
    from scipy.sparse import csr_matrix

    data = iris
    options = {}
    if mode == "sparse":
        data = csr_matrix(iris)
    elif mode == "unique":
        data = np.concatenate([iris, iris[:5]])
        options["unique"] = True
    elif mode == "density":
        options["output_dens"] = True
    elif mode == "cosine":
        options["metric"] = "cosine"
    fitted = model(**options).fit(data)
    assert fitted.embedding_.shape == (data.shape[0], 2)
    assert np.isfinite(fitted.embedding_).all()
    assert np.isfinite(fitted.kernel_bias_)
    if mode == "unique":
        assert_array_equal(fitted.embedding_[:5], fitted.embedding_[-5:])
    if mode == "density":
        assert np.isfinite(fitted.rad_orig_).all()
        assert np.isfinite(fitted.rad_emb_).all()


@pytest.mark.parametrize("mean", [True, False])
def test_general_epoch_gradients_match_sampled_loss(mean):
    x = np.random.RandomState(2).normal(size=(5, 2))

    def evaluate(x, bias):
        return _bce_epoch_gradients(
            x,
            x,
            np.array([0, 1]),
            np.array([2, 3]),
            np.ones(2),
            np.zeros(2),
            0,
            1.3,
            0.9,
            bias,
            7,
            mean,
            1.4,
            np.array([12345, 67890, 54321], dtype=np.int64),
            True,
        )

    gradient, db, _, count = evaluate(x, 0.3)
    assert count == 2
    eps = 1e-5
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric = (evaluate(plus, 0.3)[2] - evaluate(minus, 0.3)[2]) / (2 * eps)
        assert_allclose(gradient[index], numeric, rtol=1e-5, atol=1e-8)
    assert_allclose(
        db,
        (evaluate(x, 0.3 + eps)[2] - evaluate(x, 0.3 - eps)[2]) / (2 * eps),
        rtol=1e-5,
    )


def test_more_mean_negatives_approach_exact_expectation():
    x = np.random.RandomState(2).normal(size=(5, 2))
    exact_gradient = np.zeros_like(x)
    exact_bias = 0.0
    for target, positive, weight in [(2, True, 1.0)] + [
        (j, False, 0.25) for j in range(1, 5)
    ]:
        delta = x[0] - x[target]
        _, coefficient, db = bce_pair(delta @ delta, 1.3, 0.9, 0.3, positive)
        exact_gradient[0] += weight * coefficient * delta
        exact_gradient[target] -= weight * coefficient * delta
        exact_bias += weight * db
    errors = {}
    for k in [1, 500]:
        errors[k] = []
        for seed in range(20):
            gradient, db, _, _ = _bce_epoch_gradients(
                x,
                x,
                np.array([0]),
                np.array([2]),
                np.ones(1),
                np.zeros(1),
                0,
                1.3,
                0.9,
                0.3,
                k,
                True,
                1.0,
                np.random.RandomState(seed).randint(100, 2**30, 3).astype(np.int64),
                True,
            )
            errors[k].append(
                np.linalg.norm(gradient - exact_gradient) + abs(db - exact_bias)
            )
    assert np.mean(errors[500]) < 0.1 * np.mean(errors[1])
