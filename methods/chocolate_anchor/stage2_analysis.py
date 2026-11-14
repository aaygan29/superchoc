"""Fixed Stage 2 analysis for PROTOCOL.md. Run only on real panel data.

Input: a CSV of individual ratings with columns
    pair_id (int, from FROZEN_PREDICTIONS.csv), rater (str), stimulus ('mixture' | 'a' | 'b'),
    pleasantness (float, 0-10), duplicate (int, 0 or 1; 1 marks the repeated anchor presentation)
Run:  PYTHONPATH=methods python -m chocolate_anchor.stage2_analysis ratings.csv
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data/candidates/FROZEN_PREDICTIONS.csv"
FROZEN_SHA256 = "7c8eebdc41303cc12ce3a638de860a19e4ce91d2168cea1198d85644897139ab"
MA_MEDIAN = 4.97
N_PERMUTATIONS = 10_000  # PROTOCOL.md P1
PERMUTATION_SEED = 20261114
MIN_ANCHOR_SPEARMAN = 0.8
MIN_PAIRS = 30
FAMILIES = ("pred_rf_ecfp", "pred_rf_desc", "pred_hgb_desc", "pred_knn_tanimoto")


def load_frozen() -> pd.DataFrame:
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise RuntimeError(f"frozen predictions changed: sha256 {digest} != {FROZEN_SHA256}")
    return pd.read_csv(FROZEN)


def excluded_raters(ratings: pd.DataFrame, anchor_ids: set[int]) -> set[str]:
    """Raters whose duplicate anchor ratings correlate below 0 with their first ratings."""
    mix = ratings[(ratings.stimulus == "mixture") & ratings.pair_id.isin(anchor_ids)]
    out: set[str] = set()
    for rater, g in mix.groupby("rater"):
        wide = g.pivot_table(index="pair_id", columns="duplicate", values="pleasantness")
        if 0 not in wide or 1 not in wide or wide.dropna().shape[0] < 3:
            raise ValueError(f"rater {rater} lacks the duplicated anchor presentations")
        w = wide.dropna()
        if spearmanr(w[0], w[1]).statistic < 0:
            out.add(str(rater))
    return out


def permutation_p_greater(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> float:
    observed = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return float((1 + np.sum(null >= observed)) / (n + 1))


def analyse(ratings: pd.DataFrame) -> dict:
    frozen = load_frozen()
    anchors = frozen[frozen.anchor]
    tests = frozen[~frozen.anchor]
    dropped = excluded_raters(ratings, set(anchors.pair_id))
    kept = ratings[~ratings.rater.astype(str).isin(dropped) & (ratings.duplicate == 0)]

    means = kept[kept.stimulus == "mixture"].groupby("pair_id").pleasantness.mean()
    test_obs = tests.set_index("pair_id").join(means.rename("observed"), how="inner")
    result: dict = {"frozen_sha256": FROZEN_SHA256, "raters_excluded": sorted(dropped),
                    "n_test_pairs_rated": int(len(test_obs))}
    if len(test_obs) < MIN_PAIRS:
        result["status"] = f"underpowered: {len(test_obs)} < {MIN_PAIRS} pairs"
        return result

    x, y = test_obs.pred_ensemble.to_numpy(), test_obs.observed.to_numpy()
    result["P1"] = {"spearman": float(spearmanr(x, y).statistic),
                    "p_one_sided": permutation_p_greater(x, y, N_PERMUTATIONS, PERMUTATION_SEED)}
    result["P1"]["significant"] = result["P1"]["p_one_sided"] < 0.05

    anchor_obs = anchors.set_index("pair_id").join(means.rename("observed"), how="inner")
    ma = pd.read_csv(ROOT / "data/hedonics/mixture_pleasantness_ma2021.csv").set_index("mixture_id")
    anchor_obs["ma_observed"] = ma.loc[anchor_obs.ma_mixture_id, "pleasantness_mix"].to_numpy()
    link_rho = float(spearmanr(anchor_obs.observed, anchor_obs.ma_observed).statistic)
    if len(anchor_obs) < 4 or link_rho < MIN_ANCHOR_SPEARMAN:
        result["P2"] = {"status": "not evaluable: scale linking unreliable", "anchor_spearman": link_rho}
    else:
        b, a = np.polyfit(anchor_obs.ma_observed, anchor_obs.observed, 1)
        threshold = a + b * MA_MEDIAN
        disliked = test_obs[test_obs.call == "disliked"]
        below = int((disliked.observed < threshold).sum())
        result["P2"] = {"anchor_spearman": link_rho, "linked_threshold": float(threshold),
                        "n_disliked_calls": int(len(disliked)), "n_below": below,
                        "p_one_sided": float(binomtest(below, len(disliked), 0.5, alternative="greater").pvalue)}

    result["P3_families"] = {f: float(spearmanr(test_obs[f], y).statistic) for f in FAMILIES}
    return result


def main() -> None:
    ratings = pd.read_csv(sys.argv[1])
    print(json.dumps(analyse(ratings), indent=2))


if __name__ == "__main__":
    main()
