"""Learned mixture model: predict a mixture's goodness from its component set.

Motivation: real olfactory mixtures are NON-ADDITIVE (single neurons show suppression /
synergy; Duchamp-Viret et al. 2003). So mean-pooling component features throws away exactly
the interaction signal that determines whether a mixture is good. A learned,
permutation-invariant set model that also sees the spread of components can capture it.

We illustrate this on a controlled non-additive oracle: a mixture's true goodness is the latent
value of the mean embedding, SUPPRESSED when the components are incoherent (high spread).
A mean-pool model cannot see the spread; a set model (mean + std + min + max pooling) can, so
the set model beats mean-pool there (0.82 vs 0.25).

HONEST SCOPE (objective self-audit, results/NEWSLATE_findings.md section (d)): this oracle result
ONLY demonstrates the oracle is non-additive. The gap is built in by construction: an ADDITIVE
oracle (suppression=0) inverts it (mean-pool 0.84, set 0.82). On clean held-out REAL human
mixture data (Snitz 2013 n=360, Ravia 2020 n=195, molecule-disjoint transfer), the set-vs-
mean-pool advantage is within the 95% CI of zero (e.g. +0.03 [-0.06, 0.12]), and a size-only
baseline recovers much of the signal. The non-additive advantage is therefore NOT supported on
real data, and this model must not be presented as beating additive pooling on real mixtures.
The human mixture datasets carry similarity ratings, not pleasantness labels, so this model
reads perceptual distinctiveness, not goodness.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class NonAdditiveMixtureOracle:
    def __init__(self, n_pool=600, dim=8, n_peaks=6, noise_sd=0.02,
                 suppression=0.8, seed=0):
        rng = np.random.default_rng(seed)
        self.pool = rng.normal(0, 1, size=(n_pool, dim))
        self._c = rng.normal(0, 1, size=(n_peaks, dim))
        self._w = rng.uniform(0.6, 1.0, size=n_peaks)
        self._width = 0.9 * np.sqrt(dim)
        self.suppression = suppression
        self.noise_sd = noise_sd
        self._rng = rng

    def _latent(self, x):
        x = np.atleast_2d(x)
        d2 = ((x[:, None, :] - self._c[None]) ** 2).sum(2)
        return (self._w * np.exp(-d2 / (2 * self._width ** 2))).sum(1) / self._w.sum()

    def true_mixture(self, combo):
        members = self.pool[list(combo)]
        base = float(self._latent(members.mean(0))[0])
        spread = float(members.std(0).mean())          # incoherence
        return base * (1.0 - self.suppression * min(spread, 1.0))

    def query_mixture(self, combo):
        return self.true_mixture(combo) + float(self._rng.normal(0, self.noise_sd))


def mean_pool_features(oracle, combos):
    return np.vstack([oracle.pool[list(c)].mean(0) for c in combos])


def set_features(oracle, combos):
    rows = []
    for c in combos:
        m = oracle.pool[list(c)]
        rows.append(np.concatenate([m.mean(0), m.std(0), m.min(0), m.max(0)]))
    return np.vstack(rows)


def _make_combos(n_pool, k, n, rng):
    return [tuple(rng.choice(n_pool, size=k, replace=False)) for _ in range(n)]


def compare(seed=0, k=4, n_combos=1500):
    oracle = NonAdditiveMixtureOracle(seed=seed)
    rng = np.random.default_rng(seed + 1)
    combos = _make_combos(oracle.pool.shape[0], k, n_combos, rng)
    y = np.array([oracle.query_mixture(c) for c in combos])

    out = {}
    for name, feat in [("mean_pool_baseline", mean_pool_features),
                       ("learned_set_model", set_features)]:
        X = feat(oracle, combos)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        reg = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        out[name] = float(spearmanr(yte, reg.predict(Xte)).correlation)
    out["set_minus_meanpool"] = out["learned_set_model"] - out["mean_pool_baseline"]
    return out
