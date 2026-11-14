"""Tests for the post-registration exploratory analyses (fast: no full run)."""

import math

import numpy as np
import pytest

from chocolate_anchor import exploratory as X


def test_spearman_brown():
    assert X.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert X.spearman_brown(1.0) == 1.0 and X.spearman_brown(0.0) == 0.0


def test_conformal_quantile_matches_definition():
    s = np.arange(1.0, 10.0)  # n = 9
    assert X.conformal_quantile(s, 0.2) == 8.0  # ceil(10*0.8)/9 = 8/9 -> 8th smallest
    assert X.conformal_quantile(s, 0.05) == math.inf  # ceil(10*0.95)/9 > 1


def test_ecdf_rank_midrank_and_range():
    ref = np.array([1.0, 2.0, 2.0, 3.0])
    r = X.ecdf_rank(ref, np.array([0.0, 2.0, 9.0]))
    assert r[0] == 0.0 and r[2] == 1.0 and r[1] == pytest.approx(0.5)


def test_conformal_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cov = []
    for _ in range(300):
        x = rng.normal(size=400); y = 0.5 * x + rng.normal(size=400) * 0.9
        cal, te = slice(0, 200), slice(200, 400)
        r = X.conformal_rep(y[cal], x[cal], y[te], x[te], np.ones(200, bool), 0.2, 0.0)
        cov.append(r["marginal"]["coverage"])
    assert 0.77 < np.mean(cov) < 0.85


def test_mondrian_class_conditional_coverage():
    rng = np.random.default_rng(1)
    cl, cd = [], []
    for _ in range(300):
        x = rng.normal(size=600); y = 0.6 * x + rng.normal(size=600) * 0.8
        r = X.conformal_rep(y[:300], x[:300], y[300:], x[300:], np.ones(300, bool), 0.2, 0.0)["mondrian"]
        cl.append(r["coverage_liked"]); cd.append(r["coverage_disliked"])
    assert np.mean(cl) > 0.77 and np.mean(cd) > 0.77


def test_molecule_disjoint_split_has_no_shared_molecule():
    import pandas as pd
    pred = pd.read_csv(X.PRED_CSV)
    cids = pred.cids.str.split(";", expand=True).astype(int).to_numpy()
    mols = np.unique(cids)
    half1 = set(np.random.default_rng(3).permutation(mols)[: len(mols) // 2].tolist())
    in1 = np.isin(cids, list(half1))
    cal, test = in1.all(axis=1), (~in1).all(axis=1)
    assert set(cids[cal].ravel()).isdisjoint(set(cids[test].ravel()))
    assert cal.sum() + test.sum() < len(pred)  # straddlers are dropped


def test_gate_boundary_exact():
    # liquid/solid: logP > -1.72*logVP - 9.10 ; at logVP=-5 the bound is -0.5
    assert X.odorous_side(-0.4, -5.0)[0] and not X.odorous_side(-0.6, -5.0)[0]
    # gas (VP >= 760 mmHg): logP < -1.61*logVP + 8.17 ; at logVP=3 the bound is 3.34
    assert X.odorous_side(3.3, 3.0)[0] and not X.odorous_side(3.4, 3.0)[0]


def test_is_likely_odorous_known_cases_and_validation():
    ok, reason = X.is_likely_odorous("CCO", vapor_pressure_mmhg=59.0)
    assert ok and "supplied VP" in reason
    ok, reason = X.is_likely_odorous("OCC(O)C(O)C(O)C(O)CO", vapor_pressure_mmhg=1e-14)  # sorbitol-like, essentially involatile
    assert not ok
    with pytest.raises(ValueError):
        X.is_likely_odorous("not a smiles", vapor_pressure_mmhg=1.0)
    with pytest.raises(ValueError):
        X.is_likely_odorous("CCO", vapor_pressure_mmhg=0.0)


def test_power_type_one_error_and_monotonicity():
    mixtures, _, _ = X.load_rater_data()
    vc = X.fit_variance_components(mixtures)
    X.N_POWER_SIMS = 1500
    crit = X.spearman_critical(30, 0.05, 1)
    null = X.simulate_cell(vc, 0.0, 30, 30, [0.4], crit, 5)
    assert 0.02 < null["power_spearman"] < 0.09  # nominal 0.05
    lo = X.simulate_cell(vc, 0.3, 30, 30, [0.4], crit, 6)["power_spearman"]
    hi = X.simulate_cell(vc, 0.6, 30, 30, [0.4], crit, 6)["power_spearman"]
    assert hi > lo


def test_binom_critical():
    assert X.binom_critical(10, 0.05) == 9  # P(X>=9)=0.0107, P(X>=8)=0.0547
    assert X.binom_critical(3, 0.05) == 4  # unreachable
