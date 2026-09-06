"""The 'goodness' model: learn which molecular embeddings taste good.

Trains a supervised regressor mapping a molecule's flavor embedding to its
deliciousness ("goodness"). This is Criterion 1's forward model in miniature. On real
data the input is an ECFP4 fingerprint (see featurize.py) or a POM/GNN embedding; here
it is the synthetic FlavorSpace embedding so the whole pipeline can be validated against
a known oracle.

The model exposes .predict(X) used by the generator (compose.py) to score candidate
molecule combos. A classifier variant (good vs not, thresholded at a reference) is also
provided for the "identify properties as good" framing.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class GoodnessModel:
    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.reg = RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1, max_depth=None
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GoodnessModel":
        self.reg.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return self.reg.predict(np.atleast_2d(X))

    def is_good(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.predict(X) >= threshold


def build_training_set(space, n_train: int, seed: int = 0):
    """Sample molecules from the pool and observe their (noisy) goodness = labels."""
    rng = np.random.default_rng(seed)
    n = space.pool.shape[0]
    idx = rng.choice(n, size=min(n_train, n), replace=False)
    X = space.pool[idx]
    y = np.array([space.query(int(i)) for i in idx])
    return X, y, idx


def evaluate_model(space, n_train=400, seed=0):
    """Train/test split by molecule; report held-out R2 against the true surface."""
    X, y, train_idx = build_training_set(space, n_train, seed=seed)
    model = GoodnessModel(seed=seed).fit(X, y)
    all_idx = np.arange(space.pool.shape[0])
    test_idx = np.setdiff1d(all_idx, train_idx)
    y_true = space.true_pool[test_idx]            # noiseless truth for honest scoring
    y_pred = model.predict(space.pool[test_idx])
    return model, float(r2_score(y_true, y_pred))
