"""Tests for the frozen prospective prediction set (two full runs into temp dirs)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from chocolate_anchor import mixture_hedonics as mh
from chocolate_anchor import prospective as P


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = []
    for i in range(2):
        d = tmp_path_factory.mktemp(f"run{i}")
        res = P.run(candidates_dir=d / "c", results_dir=d / "r", n_jobs=2, shard_size=50)
        out.append((d, res))
    return out


def test_zero_overlap_between_training_and_ma_or_pool():
    ma, pool = mh.load_ma_molecules(), P.load_pool()
    cids = set(ma.cid.tolist()) | set(pool.table.cid.tolist())
    smi = set(ma.stripped) | set(pool.table.stripped)
    tr = mh.load_keller_training(cids, smi)  # asserts zero residual overlap itself
    assert not (set(tr.cids.tolist()) & cids)
    assert tr.n_removed_total > 0


def test_deterministic_bytes(runs):
    (d0, r0), (d1, r1) = runs
    for f in ("FROZEN_PREDICTIONS.csv", "FROZEN_COMPONENT_PREDICTIONS.csv", "scored_pairs.csv", "FROZEN_PREDICTIONS.sha256"):
        assert (d0 / "c" / f).read_bytes() == (d1 / "c" / f).read_bytes()
    assert r0["frozen_predictions_sha256"] == r1["frozen_predictions_sha256"]


def test_selection_constraints(runs):
    d, res = runs[0]
    fz = pd.read_csv(d / "c" / "FROZEN_PREDICTIONS.csv", keep_default_na=False)
    sp = pd.read_csv(d / "c" / "scored_pairs.csv", keep_default_na=False)
    pro, anc = fz[~fz.anchor], fz[fz.anchor]
    assert len(fz) == 48 and len(pro) == 40 and len(anc) == 8
    assert pro.pair_id.is_unique
    assert (pro.groupby("prediction_quintile").size() == 8).all() and pro.prediction_quintile.nunique() == 5
    assert pd.concat([pro.cid_a, pro.cid_b]).value_counts().max() <= 3
    assert sorted(anc.anchor_decile) == [1, 2, 3, 4, 7, 8, 9, 10]
    pool = P.load_pool().table.set_index("cid")
    for r in pro.itertuples():
        assert pool.eligible[r.cid_a] and pool.eligible[r.cid_b]
        assert "cocoa_key" in (r.role_a, r.role_b)
    assert len(sp) == res["counts"]["pairs_scored"] == 693
    vc = sp.prediction_quintile.value_counts()
    assert vc.max() - vc.min() <= 1
    assert not {c for c in fz.columns if "conformal" in c or c == "call"}
    comp = pd.read_csv(d / "c" / "FROZEN_COMPONENT_PREDICTIONS.csv")
    assert set(comp.cid) == set(pro.cid_a) | set(pro.cid_b) and comp.cid.is_unique
    assert len(comp.columns) == 3 + 4 + 1 + 1
    lines = (d / "c" / "FROZEN_PREDICTIONS.sha256").read_text().splitlines()
    assert [l.split("  ")[1] for l in lines] == ["FROZEN_PREDICTIONS.csv", "FROZEN_COMPONENT_PREDICTIONS.csv"]
    for l in lines:
        sha, name = l.split("  ")
        assert sha == hashlib.sha256((d / "c" / name).read_bytes()).hexdigest()
    assert "damascenone" not in " ".join(pro.name_a) + " ".join(pro.name_b)


def test_score_pairs_sharding_matches_direct():
    rng = np.random.default_rng(0)
    comp = P.ComponentTable(rng.random(20), rng.random((20, 4)))
    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    a = P.score_pairs(comp, pairs, n_jobs=2, shard_size=7)
    assert np.allclose(a[:, 0], [(comp.ensemble[i] + comp.ensemble[j]) / 2 for i, j in pairs])
    t = P.score_pairs(comp, [(0, 1, 2)], n_jobs=1, shard_size=3)  # ternary
    assert t[0, 0] == pytest.approx(comp.ensemble[:3].mean())
