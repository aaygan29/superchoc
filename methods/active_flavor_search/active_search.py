"""Online (sequential) active search over flavor space.

The technique: instead of screening molecules in a fixed batch, we choose the
next molecule to taste using everything tasted so far. We keep a Bayesian linear
model of deliciousness as a function of the flavor embedding, and pick the next
query with an acquisition rule that trades off exploiting known-good regions
against exploring uncertain ones.

Design choices and why:
- Bayesian linear regression on the embedding: closed-form posterior mean and
  variance, no external ML dependencies, cheap to update after every single
  tasting. This is the natural "online" model. Real embeddings (POM/GNN) are
  nonlinear, but the embedding is exactly the space in which perception is
  designed to be locally linear (see Qian et al. 2023), so a linear model on a
  good embedding is a defensible first method and a fair baseline to improve on.
- Two acquisition rules are provided: Upper Confidence Bound (UCB) and Thompson
  sampling. Both are standard online-decision methods with regret guarantees.

Only numpy is required.

Note on features: a linear model on the raw embedding cannot represent a bumpy
deliciousness surface, so it degrades to random search. We therefore lift the
embedding into Random Fourier Features (Rahimi & Recht, 2007), which make
Bayesian linear regression approximate a Gaussian process with an RBF kernel
while staying closed-form and online-updatable. This is the fix that makes the
method actually learn structure in flavor space.
"""

from __future__ import annotations

import numpy as np


class RandomFourierFeatures:
    """RFF map approximating an RBF kernel: phi(x) = sqrt(2/D) cos(Wx + b)."""

    def __init__(self, in_dim: int, n_features: int = 200, lengthscale: float = 3.0,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / lengthscale, size=(n_features, in_dim))
        self.b = rng.uniform(0.0, 2 * np.pi, size=n_features)
        self.scale = np.sqrt(2.0 / n_features)
        self.out_dim = n_features

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.scale * np.cos(X @ self.W.T + self.b)


class BayesianLinearModel:
    """Bayesian linear regression with a Gaussian prior. Online-updatable.

    Posterior over weights w given design X, targets y, noise var s2, prior
    precision lambda: covariance A^{-1} with A = lambda*I + X^T X / s2.
    """

    def __init__(self, dim: int, noise_var: float = 0.01, prior_precision: float = 1.0):
        self.dim = dim
        self.noise_var = noise_var
        self._A = prior_precision * np.eye(dim)
        self._b = np.zeros(dim)          # = sum_i x_i y_i / s2
        self._dirty = True
        self._cov = None
        self._mean = None

    def update(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        self._A += np.outer(x, x) / self.noise_var
        self._b += x * y / self.noise_var
        self._dirty = True

    def _refresh(self) -> None:
        if self._dirty:
            self._cov = np.linalg.inv(self._A)
            self._mean = self._cov @ self._b
            self._dirty = False

    def predict(self, X: np.ndarray):
        """Return (mean, sd) predictions for rows of X."""
        self._refresh()
        X = np.atleast_2d(X)
        mean = X @ self._mean
        var = np.einsum("ij,jk,ik->i", X, self._cov, X)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def sample_weights(self, rng: np.random.Generator) -> np.ndarray:
        self._refresh()
        # Cholesky sampling with jitter is numerically stable where a direct
        # multivariate_normal SVD can fail on near-singular high-dim covariances.
        cov = self._cov
        d = cov.shape[0]
        jitter = 1e-9
        for _ in range(6):
            try:
                L = np.linalg.cholesky(cov + jitter * np.eye(d))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            L = np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))
        return self._mean + L @ rng.standard_normal(d)


def _acquire(model, pool, tried_mask, rng, strategy="ucb", beta=0.1):
    """Score the pool and return the index of the next molecule to query.

    beta is small by design: the RFF surrogate is accurate on the pool interior but
    has large predictive variance in low-density (low-deliciousness) tails, so a big
    exploration bonus chases garbage. Validation (docs) confirmed small beta wins.
    """
    if strategy == "ucb":
        mean, sd = model.predict(pool)
        score = mean + beta * sd
    elif strategy == "thompson":
        w = model.sample_weights(rng)
        score = pool @ w
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    score = score.copy()
    score[tried_mask] = -np.inf   # never re-query the same molecule
    return int(np.argmax(score))


def run_active_search(space, budget=60, n_seed=15, strategy="ucb", beta=0.1,
                      n_features=200, lengthscale=3.0, seed=0):
    """Run the online loop against a ground-truth ``space``.

    Returns dict with the running best observed value per step and the indices
    queried. ``space`` must expose ``.pool`` (n, dim) and ``.query(idx)->float``.
    """
    rng = np.random.default_rng(seed)
    raw_pool = space.pool
    n, raw_dim = raw_pool.shape
    fmap = RandomFourierFeatures(raw_dim, n_features=n_features,
                                 lengthscale=lengthscale, seed=seed)
    pool = fmap.transform(raw_pool)          # search operates in feature space
    dim = fmap.out_dim
    tried = np.zeros(n, dtype=bool)
    model = BayesianLinearModel(dim=dim)

    best_so_far = []
    queried = []
    best = -np.inf

    # Seed with a few random tastings so the model is identifiable.
    seed_idx = rng.choice(n, size=min(n_seed, n), replace=False)
    for idx in seed_idx:
        y = space.query(int(idx))
        model.update(pool[idx], y)
        tried[idx] = True
        queried.append(int(idx))
        best = max(best, y)
        best_so_far.append(best)

    # Online phase.
    for _ in range(budget - len(seed_idx)):
        idx = _acquire(model, pool, tried, rng, strategy=strategy, beta=beta)
        y = space.query(idx)
        model.update(pool[idx], y)
        tried[idx] = True
        queried.append(idx)
        best = max(best, y)
        best_so_far.append(best)

    return {"best_so_far": np.array(best_so_far), "queried": queried}


def run_random_search(space, budget=60, seed=0):
    """Baseline: taste ``budget`` random molecules, track running best."""
    rng = np.random.default_rng(seed)
    n = space.pool.shape[0]
    order = rng.choice(n, size=min(budget, n), replace=False)
    best_so_far = []
    best = -np.inf
    for idx in order:
        y = space.query(int(idx))
        best = max(best, y)
        best_so_far.append(best)
    return {"best_so_far": np.array(best_so_far), "queried": [int(i) for i in order]}
