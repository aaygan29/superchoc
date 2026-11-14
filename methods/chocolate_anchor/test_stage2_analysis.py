"""Checks the fixed Stage 2 analysis on simulated ratings (never on real data)."""

import numpy as np
import pandas as pd
import pytest

from chocolate_anchor import stage2_analysis as S


def simulate(signal: float, n_raters: int = 30, seed: int = 0) -> pd.DataFrame:
    frozen = S.load_frozen()
    ma = pd.read_csv(S.ROOT / "data/hedonics/mixture_pleasantness_ma2021.csv").set_index("mixture_id")
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in frozen.iterrows():
        if r.anchor:
            truth = ma.loc[r.ma_mixture_id, "pleasantness_mix"]
        else:
            truth = 5 + signal * 4 * (r.pred_ensemble - 0.4) + rng.normal(0, 0.6)
        for rater in range(n_raters):
            reps = (0, 1) if r.anchor else (0,)
            for dup in reps:
                rows.append({"pair_id": r.pair_id, "rater": f"r{rater}", "stimulus": "mixture",
                             "pleasantness": truth + rng.normal(0, 1.5), "duplicate": dup})
    return pd.DataFrame(rows)


def test_detects_signal_and_links_scale():
    out = S.analyse(simulate(signal=1.0))
    assert out["P1"]["significant"]
    assert out["P2"]["anchor_spearman"] >= S.MIN_ANCHOR_SPEARMAN


def test_null_signal_rarely_significant():
    sig = [S.analyse(simulate(signal=0.0, seed=s))["P1"]["significant"] for s in range(5)]
    assert sum(sig) <= 1


def test_rejects_changed_frozen_file(monkeypatch):
    monkeypatch.setattr(S, "FROZEN_SHA256", "0" * 64)
    with pytest.raises(RuntimeError):
        S.load_frozen()
