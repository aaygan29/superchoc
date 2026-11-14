"""Checks the fixed Stage 2 analysis on simulated ratings (never on real data)."""

import numpy as np
import pandas as pd
import pytest

from chocolate_anchor import stage2_analysis as S


def simulate(signal: float, n_raters: int = 30, seed: int = 0) -> pd.DataFrame:
    pairs = S.load_frozen("FROZEN_PREDICTIONS.csv")
    molecules = S.load_frozen("FROZEN_COMPONENT_PREDICTIONS.csv")
    ma = pd.read_csv(S.ROOT / "data/hedonics/mixture_pleasantness_ma2021.csv").set_index("mixture_id")
    rng = np.random.default_rng(seed)
    rows = []

    def add(kind: str, sid: str, truth: float) -> None:
        for rater in range(n_raters):
            rows.append({"rater": f"r{rater}", "stimulus": kind, "stimulus_id": sid,
                         "pleasantness": truth + rng.normal(0, 1.5)})

    for _, r in pairs.iterrows():
        truth = (ma.loc[r.ma_mixture_id, "pleasantness_mix"] if r.anchor
                 else 5 + signal * 4 * (r.pred_ensemble - 0.4) + rng.normal(0, 0.6))
        add("pair", str(r.pair_id), truth)
    for _, r in molecules.iterrows():
        add("molecule", str(r.cid), 5 + signal * 4 * (r.pred_ensemble - 0.4) + rng.normal(0, 0.6))
    return pd.DataFrame(rows)


def test_detects_strong_signal():
    out = S.analyse(simulate(signal=1.0))
    assert out["P1"]["supported"] and out["P2"]["supported"]
    assert out["descriptive"]["anchors_spearman_with_ma"] > 0.8


def test_null_rarely_supported():
    supported = [S.analyse(simulate(signal=0.0, seed=s))["P1"]["supported"] for s in range(5)]
    assert sum(supported) <= 1


def test_rejects_changed_frozen_file(monkeypatch):
    monkeypatch.setitem(S.FROZEN_SHA256, "FROZEN_PREDICTIONS.csv", "0" * 64)
    with pytest.raises(RuntimeError):
        S.load_frozen("FROZEN_PREDICTIONS.csv")
