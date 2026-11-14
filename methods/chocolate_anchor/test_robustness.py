"""Tests for the exploratory robustness module (cheap; the full run is exercised with small settings)."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from chocolate_anchor import robustness as R


def _design(n_mol=12, n_pairs=30, seed=0):
    rng = np.random.default_rng(seed)
    pairs = np.array([rng.choice(n_mol, 2, replace=False) for _ in range(n_pairs)])
    return pairs[:, 0], pairs[:, 1], n_mol


def test_simple_descriptors_shape_and_sulfur():
    x = R.simple_descriptors([Chem.MolFromSmiles(s) for s in ("CCS", "c1ccccc1", "CSSC")])
    assert x.shape == (3, len(R.DESCRIPTORS))
    assert list(x[:, -1]) == [1, 0, 2]
    assert x[1, 6] == 1  # ring count


def test_vectorized_bootstrap_matches_tie_safe_loop():
    a, b, n_mol = _design()
    rng = np.random.default_rng(1)
    pred, obs = rng.normal(size=len(a)), rng.normal(size=len(a))
    counts = R._draw_counts(np.random.default_rng(2), 200, n_mol)
    ia, ib = a, b
    loop = R._boot_rhos_loop(pred, obs, ia, ib, counts)
    vec = R._boot_rhos_vectorized(pred, obs, ia, ib, counts)
    assert np.allclose(loop, vec, equal_nan=True)


def test_lower_bound_separates_signal_from_noise():
    a, b, _ = _design(n_mol=20, n_pairs=40)
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    strong = R.cluster_bootstrap_lower_bound(x, x + 0.1 * rng.normal(size=40), a, b, n_boot=500)
    noise = R.cluster_bootstrap_lower_bound(x, rng.normal(size=40), a, b, n_boot=500)
    assert strong > 0.8 and noise < 0.3 and strong > noise


def test_lower_bound_is_deterministic_and_level_monotone():
    a, b, _ = _design()
    rng = np.random.default_rng(4)
    p, o = rng.normal(size=len(a)), rng.normal(size=len(a))
    assert R.cluster_bootstrap_lower_bound(p, o, a, b, n_boot=300) == R.cluster_bootstrap_lower_bound(p, o, a, b, n_boot=300)
    assert R.cluster_bootstrap_lower_bound(p, o, a, b, level=0.99, n_boot=300) <= R.cluster_bootstrap_lower_bound(p, o, a, b, level=0.9, n_boot=300)


def test_simulation_type_one_error_is_low_and_power_ordered():
    a, b, _ = _design(n_mol=40, n_pairs=40)
    vc = {"mu_hat": np.random.default_rng(5).normal(size=198), "resid": np.random.default_rng(6).normal(size=5000), "shrink": 0.8}
    crit = 0.27
    null = R.simulate_p1_rule(a, b, 0.0, vc, 0.0, crit, n_sims=300, n_boot=200, seed=7)
    strong = R.simulate_p1_rule(a, b, 0.9, vc, 0.95, crit, n_sims=300, n_boot=200, seed=7)
    assert null["p_permutation_and_cluster_lb"] <= null["p_permutation_alone"] < 0.15
    assert strong["p_permutation_and_cluster_lb"] > null["p_permutation_and_cluster_lb"]


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("rob")
    r = R.run(n_boot=100, n_jobs=1, simulate=False, out_json=d / "r.json", out_md=d / "R.md")
    return r, d / "R.md"


def test_run_structure(small_run):
    r, md = small_run
    assert r["n_mixtures"] == 198 and r["training_registered"] == 376 and r["training_prospective_model"] == 358
    assert r["a_baselines"]["ensemble"]["estimate"] == pytest.approx(0.4896, abs=1e-3)  # registered H2
    for k in ("ridge", "rf8"):
        assert -1 < r["a_baselines"][k]["estimate"] < 1
    assert r["b_near_duplicate"]["tanimoto_lt_0.9"]["n_mixtures"] >= r["b_near_duplicate"]["tanimoto_lt_0.7"]["n_mixtures"] > 0
    assert r["c_prospective_model_on_ma"]["spearman_with_registered_model_mixture_predictions"] > 0.5
    assert len(md.read_text().splitlines()) <= 50
