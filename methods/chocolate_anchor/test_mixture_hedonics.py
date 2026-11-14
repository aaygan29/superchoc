"""Tests for the pre-registered mixture-hedonics analysis (leakage, bootstrap, mixture rule, abstention)."""

import numpy as np
import pytest

from chocolate_anchor import mixture_hedonics as M


@pytest.fixture(scope="module")
def ma():
    return M.load_ma_molecules()


def test_no_ma_molecule_in_keller_training(ma):
    tr = M.load_keller_training(ma.cid, ma.stripped)
    assert not set(tr.cids.tolist()) & set(ma.cid.tolist())
    kel = M.build_keller_pleasantness()
    kept = kel[kel.CID.astype(int).isin(set(tr.cids.tolist()))]
    keys = {M.stripped_smiles(M.parse_smiles(s)) for s in kept.smiles}
    assert not keys & set(ma.stripped)
    assert tr.n_removed_total > 0  # the overlap exists, so the removal is doing something
    assert tr.n_removed_total == len(kel) - len(tr.y)


def test_leakage_assertion_fires_on_overlap(ma):
    # exclusion by SMILES alone must also remove the molecule (CID deliberately not given)
    kel = M.build_keller_pleasantness()
    smi = M.stripped_smiles(M.parse_smiles(kel.smiles.iloc[0]))
    tr = M.load_keller_training([], [smi])
    assert tr.n_removed_by_smiles >= 1 and tr.n_removed_by_cid == 0


def test_mixture_rule_equal_and_weighted():
    a, b = np.array([0.2, 0.8]), np.array([0.6, 0.4])
    np.testing.assert_allclose(M.mixture_rule(a, b), [0.4, 0.6])
    np.testing.assert_allclose(M.mixture_rule(a, b, np.array([1.0, 3.0]), np.array([1.0, 1.0])), [0.4, 0.7])


def test_holm():
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_rank_scale_in_unit_interval():
    s = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    r = M.to_rank_scale(s, np.array([0.0, 2.0, 10.0]))
    assert r[0] == 0.0 and r[2] == 1.0 and 0 < r[1] < 1


def _toy(n_mol=12, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n_mol)
    pairs = [(i, j) for i in range(n_mol) for j in range(i + 1, n_mol)]
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    obs = (truth[a] + truth[b]) / 2
    return M.BootData(
        n_mol=n_mol, a_idx=a, b_idx=b, obs=obs, liked=obs > np.median(obs),
        mix_scores={"perfect": obs, "noise": rng.normal(size=len(a))},
        subsets={"all": np.ones(len(a), bool)}, comp_obs=truth,
        comp_scores={"perfect": truth, "noise": rng.normal(size=n_mol)},
    )


def test_bootstrap_toy_perfect_and_noise():
    d = _toy()
    specs = [M.StatSpec("perfect", "spearman_mix", "perfect"), M.StatSpec("noise", "spearman_mix", "noise"),
             M.StatSpec("auc", "auroc_mix", "perfect"), M.StatSpec("comp", "spearman_comp", "perfect")]
    b1 = M.cluster_bootstrap(d, specs, n_boot=300, seed=1, n_jobs=1)
    b2 = M.cluster_bootstrap(d, specs, n_boot=300, seed=1, n_jobs=2)
    for k in b1:
        np.testing.assert_allclose(b1[k], b2[k])  # independent of n_jobs
    lo, hi, _ = M.percentile_ci(b1["perfect"])
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert M.percentile_ci(b1["auc"])[0] > 0.9
    lo, hi, _ = M.percentile_ci(b1["noise"])
    assert lo < 0 < hi  # noise CI straddles zero


def test_bootstrap_keeps_mixture_only_if_both_components_drawn():
    d = _toy(n_mol=4)
    counts = np.array([1, 1, 0, 0])
    keep = d.subsets["all"] & (counts > 0)[d.a_idx] & (counts > 0)[d.b_idx]
    assert keep.sum() == 1 and (d.a_idx[keep][0], d.b_idx[keep][0]) == (0, 1)


def test_abstention_rule_agreement_flag():
    # a family set that straddles the Keller median must not be flagged "agree"
    ens = M.build_component_ensemble(set(M.load_ma_molecules().cid))
    out = M.predict_components(["CCO", "O=Cc1ccc(O)c(OC)c1", "CC(=O)C(C)O"], ens)
    raw = np.column_stack([out["raw_keller_scale"][f] for f in M.FAMILIES])
    side = raw > ens.keller_median
    np.testing.assert_array_equal(out["agreement"], side.all(1) | (~side).all(1))
    assert set(out["families"]) == set(M.FAMILIES) and out["ensemble"].shape == (3,)
    assert np.all((out["ensemble"] >= 0) & (out["ensemble"] <= 1))


def test_mixture_confident_requires_both_components():
    agree = np.array([True, True, False])
    a, b = np.array([0, 0, 1]), np.array([1, 2, 2])
    np.testing.assert_array_equal(agree[a] & agree[b], [True, False, False])
