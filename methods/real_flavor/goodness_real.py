"""Real goodness model: predict human panel pleasantness from molecular structure.

Trains on the Keller 2016 real labels (see real_data.py), features = ECFP4. This is
Criterion 1 on real data. We report cross-validated Spearman and R2, the standard way the
DREAM/olfaction literature scores structure->pleasantness (pleasantness is among the more
predictable perceptual attributes; Keller et al. 2017, Li et al. 2018).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class RealGoodnessModel:
    def __init__(self, n_estimators=300, seed=0):
        self.reg = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                         n_jobs=-1)
        self._fitted = False

    def fit(self, X, y):
        self.reg.fit(X, y); self._fitted = True; return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("fit first")
        return self.reg.predict(np.atleast_2d(X))


def cross_validate(X, y, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        m = RealGoodnessModel(seed=seed).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    rho = float(spearmanr(y, preds).correlation)
    r2 = float(r2_score(y, preds))
    return {"spearman": round(rho, 4), "r2": round(r2, 4), "n": int(len(y))}
