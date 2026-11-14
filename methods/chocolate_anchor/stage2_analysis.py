"""Fixed Stage 2 analysis for PROTOCOL.md. Run only on real panel data.

Input: a CSV of individual ratings with columns
    rater (str), stimulus ('pair' | 'molecule'), stimulus_id (str: pair_id from
    FROZEN_PREDICTIONS.csv, or the cid from FROZEN_COMPONENT_PREDICTIONS.csv),
    pleasantness (float)
Run:  PYTHONPATH=methods python -m chocolate_anchor.stage2_analysis ratings.csv
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chocolate_anchor.robustness import cluster_bootstrap_lower_bound

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "data/candidates"
FROZEN_SHA256 = {
    "FROZEN_PREDICTIONS.csv": "b0d9035759410da49bb590f8a0c43d06a17f6fe6797c2570176ddb7eb0e7f9dc",
    "FROZEN_COMPONENT_PREDICTIONS.csv": "833236be0aabedd343d181e2c48fc9e677c70350d9db4a0ce0dc149be6e024f9",
}
N_PERMUTATIONS = 10_000  # PROTOCOL.md P1, P2
PERMUTATION_SEED = 20261114
BOOT_LEVEL = 0.95  # one-sided lower bound, PROTOCOL.md P1
ALPHA = 0.05
FAMILIES = ("pred_rf_ecfp", "pred_rf_desc", "pred_hgb_desc", "pred_knn_tanimoto")


def load_frozen(name: str) -> pd.DataFrame:
    path = CANDIDATES / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256[name]:
        raise RuntimeError(f"{name} changed: sha256 {digest} != {FROZEN_SHA256[name]}")
    return pd.read_csv(path)


def permutation_p_greater(x: np.ndarray, y: np.ndarray) -> float:
    observed = spearmanr(x, y).statistic
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERMUTATIONS)])
    return float((1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1))


def panel_means(ratings: pd.DataFrame, stimulus: str) -> pd.Series:
    sub = ratings[ratings.stimulus == stimulus].assign(stimulus_id=lambda d: d.stimulus_id.astype(str))
    return sub.groupby("stimulus_id").pleasantness.mean().rename("observed")


def analyse(ratings: pd.DataFrame) -> dict:
    pairs = load_frozen("FROZEN_PREDICTIONS.csv")
    molecules = load_frozen("FROZEN_COMPONENT_PREDICTIONS.csv")
    pair_means = panel_means(ratings, "pair")
    result: dict = {"frozen_sha256": FROZEN_SHA256}

    tests = pairs[~pairs.anchor].set_index("pair_id").join(pair_means, how="inner")
    x, y = tests.pred_ensemble.to_numpy(), tests.observed.to_numpy()
    p_perm = permutation_p_greater(x, y)
    lower = cluster_bootstrap_lower_bound(x, y, tests.cid_a.to_numpy(), tests.cid_b.to_numpy(), level=BOOT_LEVEL)
    result["P1"] = {"n_pairs": int(len(tests)), "spearman": float(spearmanr(x, y).statistic),
                    "permutation_p_one_sided": p_perm, "cluster_bootstrap_lower_bound": lower,
                    "supported": bool(p_perm < ALPHA and lower > 0)}

    mol = molecules.assign(cid=molecules.cid.astype(str)).set_index("cid").join(panel_means(ratings, "molecule"), how="inner")
    mx, my = mol.pred_ensemble.to_numpy(), mol.observed.to_numpy()
    p2 = permutation_p_greater(mx, my)
    result["P2"] = {"n_molecules": int(len(mol)), "spearman": float(spearmanr(mx, my).statistic),
                    "permutation_p_one_sided": p2, "supported": bool(p2 < ALPHA)}

    anchors = pairs[pairs.anchor].set_index("pair_id").join(pair_means, how="inner")
    ma = pd.read_csv(ROOT / "data/hedonics/mixture_pleasantness_ma2021.csv").set_index("mixture_id")
    ma_obs = ma.loc[anchors.ma_mixture_id, "pleasantness_mix"].to_numpy()
    result["descriptive"] = {
        "families_P1_spearman": {f: float(spearmanr(tests[f], y).statistic) for f in FAMILIES},
        "anchors_spearman_with_ma": float(spearmanr(anchors.observed, ma_obs).statistic),
        "n_anchors": int(len(anchors)),
    }
    return result


def main() -> None:
    print(json.dumps(analyse(pd.read_csv(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
