"""Unit tests for the online active-search method. Run: python -m pytest -q (or python test_active_search.py)."""

import numpy as np

from ground_truth import SyntheticFlavorSpace
from active_search import BayesianLinearModel, run_active_search, run_random_search


def test_ground_truth_reproducible():
    a = SyntheticFlavorSpace(seed=7)
    b = SyntheticFlavorSpace(seed=7)
    assert np.allclose(a.pool, b.pool)
    assert np.allclose(a.true_pool, b.true_pool)
    assert 0.0 <= a.best_true <= 1.0


def test_blm_learns_linear_signal():
    # If truth is exactly linear, posterior mean should recover the weights.
    rng = np.random.default_rng(0)
    dim = 5
    w_true = rng.normal(size=dim)
    m = BayesianLinearModel(dim=dim, noise_var=1e-4, prior_precision=1e-3)
    X = rng.normal(size=(400, dim))
    for x in X:
        m.update(x, float(x @ w_true))
    mean, sd = m.predict(X[:10])
    assert np.allclose(mean, X[:10] @ w_true, atol=1e-2)
    assert np.all(sd >= 0)


def test_never_requeries():
    space = SyntheticFlavorSpace(seed=1)
    out = run_active_search(space, budget=40, seed=1)
    assert len(out["queried"]) == len(set(out["queried"]))  # all unique


def test_best_so_far_monotone():
    space = SyntheticFlavorSpace(seed=2)
    for fn in (run_active_search, run_random_search):
        out = fn(space, budget=30, seed=2)
        b = out["best_so_far"]
        assert np.all(np.diff(b) >= 0)  # running max never decreases


def test_active_beats_random_on_average():
    # Aggregate check over several seeds using validated defaults.
    ga = []
    for s in range(8):
        sa = SyntheticFlavorSpace(seed=s)
        sr = SyntheticFlavorSpace(seed=s)
        a = run_active_search(sa, budget=50, seed=s)["best_so_far"][-1]
        r = run_random_search(sr, budget=50, seed=s)["best_so_far"][-1]
        ga.append(a - r)
    assert np.mean(ga) > 0.02  # comfortably positive, not a coin-flip


def test_thompson_runs_and_is_stable():
    # Thompson sampling must not crash on the high-dim covariance (Cholesky+jitter).
    space = SyntheticFlavorSpace(seed=3)
    out = run_active_search(space, budget=30, strategy="thompson", seed=3)
    assert len(out["queried"]) == 30
    assert np.all(np.diff(out["best_so_far"]) >= 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
