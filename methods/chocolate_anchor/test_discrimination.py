"""Tests for the odor-mixture discrimination model (needs pyrfume's bundled data; no network)."""

import json

import numpy as np
import pytest

from chocolate_anchor import discrimination as D

VANILLIN, LIMONENE, ETHYL_BUTYRATE = "O=Cc1ccc(O)c(OC)c1", "CC(=C)C1CCC(C)=CC1", "CCCC(=O)OCC"
EUGENOL, ACETOIN, HEXANAL = "C=CCc1ccc(O)c(OC)c1", "CC(=O)C(C)O", "CCCCCC=O"


@pytest.fixture(scope="module")
def data():
    return D.load_bushdid()


@pytest.fixture(scope="module")
def ft(data):
    return D.build_features(data)


def test_data_reading_verified(data):
    assert len(data.triangles) == 264
    for t in data.triangles:
        assert len(t.odd_cids - t.dup_cids) == t.n_differ
        assert 0 <= t.k_correct <= D.N_SUBJECTS
    assert {t.n_components for t in data.triangles} == {1, 10, 20, 30}


def test_features_cover_all_triangles(ft):
    assert ft.audit["triangles_used"] == len(ft.triangles) == ft.X.shape[0]
    assert np.isfinite(ft.X).all()
    assert ft.audit["leffingwell_molecules_covered"] <= ft.audit["bushdid_molecules"]


def test_fit_recovers_synthetic_psychometric():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 1))
    p = D.p_correct_from_z(-0.5 + 1.5 * x[:, 0])
    k = rng.binomial(D.N_SUBJECTS, p)
    m = D.fit_model("perceptual", x, k)
    # standardised slope ~ 1.5 * std(x)
    assert m.intercept == pytest.approx(-0.5, abs=0.2)
    assert m.slopes[0] == pytest.approx(1.5 * x.std(), abs=0.3)


def test_fit_rejects_constant_feature():
    with pytest.raises(ValueError, match="constant"):
        D.fit_model("perceptual", np.ones((50, 1)), np.full(50, 10))


def test_predictions_never_below_chance(ft):
    for name in D.FEATURE_SETS:
        X = D._cols(ft, name)
        p = D.fit_model(name, X, ft.k).p_correct(X)
        assert (p >= D.CHANCE).all() and (p < 1).all()


def test_cv_runs_and_uses_every_triangle_once_per_seed(ft):
    oof = D.grouped_cv(ft, seeds=(0,), folds=10)
    assert not np.isnan(oof.preds["combined"]).any()


def test_molecule_holdout_tests_small_subset(data, ft):
    oof, info = D.molecule_holdout(ft, list(data.cid_to_smiles), seeds=(0,))
    assert 0 < info["n_test_triangles_ever"] < 100
    summary = D.summarize(ft, oof, n_boot=200)
    assert summary["n_triangles_tested"] == info["n_test_triangles_ever"]


def test_perceptual_distance_zero_for_identical_and_missing_reported():
    d, missing = D.perceptual_distance([VANILLIN, LIMONENE], [LIMONENE, VANILLIN])
    assert d == pytest.approx(0.0, abs=1e-6) and missing == []
    _, missing = D.perceptual_distance([VANILLIN], ["[Xe]"])
    assert missing == ["[Xe]"]


@pytest.fixture(scope="module")
def model():
    return D.fit_default(save=False)


def test_predict_more_overlap_is_harder(model):
    a = [VANILLIN, LIMONENE, ETHYL_BUTYRATE, EUGENOL]
    mostly_same = [VANILLIN, LIMONENE, ETHYL_BUTYRATE, HEXANAL]
    all_diff = [ACETOIN, HEXANAL, "CCCCC=O", "CCCCCCO"]
    p_same = D.predict_details(model, a, mostly_same).p_correct
    p_diff = D.predict_details(model, a, all_diff, allow_missing_descriptors=True).p_correct
    assert D.CHANCE <= p_same < p_diff <= 1.0


def test_predict_weights_change_result_and_validate(model):
    a, b = [VANILLIN, LIMONENE, EUGENOL], [VANILLIN, LIMONENE, HEXANAL]
    p0 = D.predict_details(model, a, b).p_correct
    p1 = D.predict_details(model, a, b, weights_a=[10, 1, 1]).p_correct
    assert p0 != p1
    with pytest.raises(ValueError, match="length"):
        D.predict_details(model, a, b, weights_a=[1, 1])


def test_predict_fails_loudly_on_missing_descriptors(model):
    with pytest.raises(ValueError, match="Leffingwell"):
        D.predict_details(model, [VANILLIN, "[Xe]"], [LIMONENE])
    with pytest.raises(ValueError, match="non-empty"):
        D.predict_details(model, [], [LIMONENE])


def test_model_json_roundtrip(model):
    assert D.DiscriminationModel.from_json(json.loads(json.dumps(model.to_json()))) == model


def test_saved_results_exist_and_state_comparison():
    res = json.loads((D.RESULTS_DIR / "discrimination.json").read_text())
    for part in ("grouped_cv_10fold_3seeds", "molecule_holdout_5fold_3seeds"):
        assert res[part]["models"]["perceptual"]["gain_vs_baseline_logloss"]["verdict"] in (
            "model better", "baseline better", "indistinguishable")
    assert (D.RESULTS_DIR / "discrimination_model.json").exists()
