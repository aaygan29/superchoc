"""Synthetic flavor space with mixture support (superset of the active-search version).

Same construction as methods/active_flavor_search/ground_truth.py (a POM-style embedding
mapped to a noisy deliciousness surface), extended with mixtures: a combo of molecules is
aggregated into a single mixture embedding, and its deliciousness is the same latent
surface evaluated at that point, plus noise.

Aggregation choice: mixture embedding = mean of member embeddings. This is a deliberate
simplification. Real mixtures show suppression and synergy (see docs/LITERATURE.md), so
mean-pooling is a first-order stand-in, documented so nobody mistakes it for the real
mixture-perception model. It is sufficient for the honest question here: can an ML
goodness model + generator compose novel combos that are genuinely good on the oracle,
not merely good in the model's opinion?
"""

from __future__ import annotations

import numpy as np


class FlavorSpace:
    def __init__(self, n_pool: int = 800, dim: int = 8, n_peaks: int = 6,
                 noise_sd: float = 0.03, seed: int = 0):
        self.dim = dim
        self.noise_sd = noise_sd
        self._rng = np.random.default_rng(seed)
        self.pool = self._rng.normal(0.0, 1.0, size=(n_pool, dim))
        self._peak_centers = self._rng.normal(0.0, 1.0, size=(n_peaks, dim))
        self._peak_weights = self._rng.uniform(0.6, 1.0, size=n_peaks)
        self._peak_width = 0.9 * np.sqrt(dim)
        self._true_pool = self._latent(self.pool)

    def _latent(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        d2 = ((x[:, None, :] - self._peak_centers[None, :, :]) ** 2).sum(axis=2)
        vals = (self._peak_weights[None, :] * np.exp(-d2 / (2 * self._peak_width ** 2))).sum(axis=1)
        return vals / self._peak_weights.sum()

    # --- single molecules ---
    def query(self, idx: int) -> float:
        return float(self._true_pool[idx] + self._rng.normal(0.0, self.noise_sd))

    @property
    def true_pool(self) -> np.ndarray:
        return self._true_pool

    # --- mixtures (combos of molecule indices) ---
    def mixture_embedding(self, indices) -> np.ndarray:
        return self.pool[list(indices)].mean(axis=0)

    def true_mixture(self, indices) -> float:
        return float(self._latent(self.mixture_embedding(indices))[0])

    def query_mixture(self, indices) -> float:
        return self.true_mixture(indices) + float(self._rng.normal(0.0, self.noise_sd))

    def chocolate_reference(self, k: int = 4, n_samples: int = 400,
                            percentile: float = 88.0, seed: int = 12345):
        """A 'chocolate' combo: excellent but deliberately NOT the maximum.

        Wintermute's conjecture is precisely that chocolate is not the peak of delight,
        so we define chocolate as a high-percentile random combo (default 88th), not the
        best-of-N. The target is then to find NOVEL combos that beat this very-good-but-
        not-maximal reference, which is the conjecture's actual claim."""
        rng = np.random.default_rng(seed)
        n = self.pool.shape[0]
        combos = [tuple(rng.choice(n, size=k, replace=False)) for _ in range(n_samples)]
        vals = np.array([self.true_mixture(c) for c in combos])
        target = np.percentile(vals, percentile)
        # the combo whose true goodness is closest to the percentile value
        j = int(np.argmin(np.abs(vals - target)))
        return {"combo": combos[j], "value": float(vals[j]),
                "embedding": self.mixture_embedding(combos[j])}
