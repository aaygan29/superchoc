"""Reproducible synthetic ground truth for validating the online flavor-search method.

This is NOT a claim about real flavor space. It is a deliberately simple, fully
controlled stand-in for the object the real method will eventually query: a
POM-style flavor embedding (see docs/LITERATURE.md) mapped to a scalar
"deliciousness" as a human panel would rate it, with noise.

We use it only to answer one honest question: does the online (sequential)
search method find high-deliciousness points using fewer panel queries than
random screening? If it cannot beat random on a controlled problem, it will not
help on the real one.
"""

from __future__ import annotations

import numpy as np


class SyntheticFlavorSpace:
    """A fixed pool of molecules in a d-dim embedding with a latent deliciousness.

    Deliciousness is a smooth mixture-of-Gaussians surface over the embedding
    (several "delicious regions", most of the pool mediocre), plus observation
    noise standing in for panel disagreement. The surface and pool are fully
    determined by ``seed`` so every validation run is reproducible.
    """

    def __init__(self, n_pool: int = 2000, dim: int = 8, n_peaks: int = 6,
                 noise_sd: float = 0.03, seed: int = 0):
        self.dim = dim
        self.noise_sd = noise_sd
        self._rng = np.random.default_rng(seed)

        # The molecular library: a fixed pool of embeddings.
        self.pool = self._rng.normal(0.0, 1.0, size=(n_pool, dim))

        # Latent deliciousness surface: a few smooth Gaussian "delight" regions.
        # Peaks sit inside the data distribution and are wide enough (relative to
        # sqrt(dim)) that deliciousness varies smoothly and learnably across the
        # pool, rather than being a vanishing-measure needle. This is what makes
        # the online-vs-random comparison a fair test.
        self._peak_centers = self._rng.normal(0.0, 1.0, size=(n_peaks, dim))
        self._peak_weights = self._rng.uniform(0.6, 1.0, size=n_peaks)
        self._peak_width = 0.9 * np.sqrt(dim)

        # Precompute noiseless truth for the pool (used only for scoring/oracle-purity).
        self._true_pool = self._latent(self.pool)

    def _latent(self, x: np.ndarray) -> np.ndarray:
        """Noiseless deliciousness for embedding rows ``x``. Range ~[0, 1]."""
        x = np.atleast_2d(x)
        # (n, peaks) squared distances
        d2 = ((x[:, None, :] - self._peak_centers[None, :, :]) ** 2).sum(axis=2)
        vals = (self._peak_weights[None, :] * np.exp(-d2 / (2 * self._peak_width ** 2))).sum(axis=1)
        return vals / self._peak_weights.sum()

    def query(self, idx: int) -> float:
        """A single 'panel tasting' of pool molecule ``idx``: noisy deliciousness."""
        return float(self._true_pool[idx] + self._rng.normal(0.0, self.noise_sd))

    # --- oracle-purity helpers: used for scoring a run, never given to the searcher ---
    @property
    def true_pool(self) -> np.ndarray:
        return self._true_pool

    @property
    def best_true(self) -> float:
        return float(self._true_pool.max())

    def chocolate_reference(self, percentile: float = 90.0) -> float:
        """A 'chocolate is very good but not the max' reference value."""
        return float(np.percentile(self._true_pool, percentile))
