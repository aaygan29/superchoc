"""Combo-pleasantness benchmark: a mixture-pleasantness target grounded in REAL single-molecule
labels plus a literature pooling law.

Honest data-availability note (checked, 2026). There is no accessible public dataset of human
HEDONIC (pleasantness) ratings for odor MIXTURES: Keller & Vosshall 2016 is single molecules;
Weiss & Sobel 2012 rates mixture IDENTIFICATION (olfactory white), not pleasantness; Snitz 2013
and Ravia 2020 rate mixture SIMILARITY / intensity, not pleasantness. So a real combo-pleasantness
dataset cannot simply be loaded. This module therefore builds the closest honest substitute: a
SEMI-SYNTHETIC benchmark whose components carry REAL Keller pleasantness, aggregated by a pooling
law taken from the mixture-perception literature. It upgrades mixture_model.py, whose oracle was
fully synthetic (an invented latent surface), to one anchored in real per-molecule labels.

It is a benchmark for the MODELING question ("given a pooling law, can a set model recover
mixture pleasantness better than mean-pooling?"), NOT a claim about real deliciousness. The
genuine gap (human panel ratings of mixtures) remains and is the right next data-collection step.

Pooling law (literature-grounded, not additive):
  base   = mean of component pleasantness                       (additive baseline)
  + dom  = pull toward the most pleasant / dominant note        (mixture dominance)
  - comp = regression toward the population mean as the number of components grows
           (Laing: humans resolve <=3-4 notes; Weiss & Sobel: many components -> bland "white")
This is deliberately NON-additive, so mean-pooling is provably limited (FlavorMath.lean best-part
bound) and a set model that sees dispersion/complexity can do better.

Refs: Laing (odor mixture analysis, <=3-4 resolvable components); Weiss & Sobel 2012 (olfactory
white); Duchamp-Viret 2003 (non-additive single-neuron mixture responses).
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from real_data import build_keller_pleasantness

_HERE = os.path.dirname(__file__)


def pooling_law(pls, pop_mean, dom=0.35, comp=0.12):
    """Literature-grounded non-additive mixture pleasantness from component pleasantness `pls`."""
    pls = np.asarray(pls, dtype=float)
    base = pls.mean()
    dominance = dom * (pls.max() - base)                     # pull toward the best/dominant note
    k = len(pls)
    complexity = comp * min(1.0, (k - 1) / 5.0) * (base - pop_mean)  # regress to mean as k grows
    return base + dominance - complexity


def build_benchmark(n=4000, k_range=(2, 8), noise_sd=6.0, seed=0):
    """Real component pleasantness + literature pooling + perceptual noise (so the target is not
    a trivially-invertible function of the summary statistics; noise_sd approximates panel
    rating noise)."""
    df = build_keller_pleasantness().reset_index(drop=True)
    pls = df["pleasantness"].to_numpy(dtype=float)
    pop_mean = float(pls.mean())
    rng = np.random.default_rng(seed)
    combos, targets = [], []
    for _ in range(n):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        idx = rng.choice(len(pls), size=k, replace=False)
        combos.append(idx)
        targets.append(pooling_law(pls[idx], pop_mean) + rng.normal(0, noise_sd))
    return pls, np.array(targets, dtype=float), combos


def _feats(pls, combos, mode):
    rows = []
    for idx in combos:
        v = pls[idx]
        if mode == "mean":
            rows.append([v.mean()])                          # additive baseline
        else:  # set features: dispersion + complexity (can capture non-additivity)
            rows.append([v.mean(), v.std(), v.min(), v.max(), len(v)])
    return np.array(rows, dtype=float)


def evaluate(seed=0):
    pls, y, combos = build_benchmark(seed=seed)
    out = {}
    for mode in ("mean", "set"):
        X = _feats(pls, combos, mode)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        pred = np.zeros_like(y)
        for tr, te in kf.split(X):
            pred[te] = RandomForestRegressor(n_estimators=200, random_state=seed,
                                             n_jobs=-1).fit(X[tr], y[tr]).predict(X[te])
        out[mode] = round(float(spearmanr(pred, y).correlation), 3)
    # pure mean-pool (no learning) as the strict additive reference
    mean_only = np.array([pls[idx].mean() for idx in combos])
    out["additive_mean_only_spearman"] = round(float(spearmanr(mean_only, y).correlation), 3)
    out["n_combos"] = len(y)
    out["reading"] = ("set model (dispersion+complexity) beats additive mean-pooling on the "
                      "real-label combo-pleasantness benchmark"
                      if out["set"] > out["mean"] else "no non-additive advantage")
    return out


if __name__ == "__main__":
    res = evaluate()
    print("combo-pleasantness benchmark (real Keller labels + literature pooling):")
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(_HERE, "results", "combo_pleasantness.json"), "w"), indent=2)
    print("wrote results/combo_pleasantness.json")
