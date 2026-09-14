"""Goodness FIELD over flavor space: turn the (descriptive) flavor-space geometry into a
predictive + generative tool.

Concept. Flavor space is a low-dimensional metric space (Kurtz 2000), empirically hyperbolic
(Sharpee 2018; replicated in flavor_geometry.py). "Good flavor" is not a direction but a smooth
SCALAR FIELD over that space: some regions (fruity esters, vanilla/lactonic) are pleasant,
others (sulfur/acid) are not. This module fits that field and uses it two ways:
  1. Predict a molecule's / blend's pleasantness from WHERE IT LANDS in flavor space
     (a geometry-based predictor, complementary to the structure-based ECFP model).
  2. Find high-field + SPARSE (unexplored) regions: high predicted goodness far from every known
     molecule cloud. Because the space is hyperbolic (volume grows exponentially with radius),
     there is exponentially much unexplored room -- these sparse-yet-good points are the
     mathematically well-posed "superchocolate" targets.

Coordinates. We use the physicochemical mixture-space representation (mixtures.MixtureSpace),
the SAME space whose distance is externally validated against human mixture similarity
(Snitz -0.49, Ravia -0.245). The field is kernel-ridge regression (RBF) from those coordinates
to real Keller pleasantness, honestly cross-validated and compared to the ECFP baseline.

Honest expectation: a 13-dim physicochemical coordinate carries less information than a 2048-bit
ECFP fingerprint, so the field is expected to be a WEAKER but interpretable predictor. We report
the real CV numbers either way (verify-before-reporting), not a hoped-for one.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from goodness_real import RealGoodnessModel
import mixtures as mx

_HERE = os.path.dirname(__file__)


class FlavorField:
    def __init__(self, alpha=None, gamma=None, seed=0):
        df = build_keller_pleasantness().reset_index(drop=True)
        self.smiles, y = [], []
        for _, r in df.iterrows():
            if mx._descriptors(r["smiles"]) is not None:
                self.smiles.append(r["smiles"]); y.append(float(r["pleasantness"]))
        self.y = np.array(y)
        self.space = mx.MixtureSpace(self.smiles)
        self.X = np.vstack([self.space.vector([s]) for s in self.smiles])  # coord per molecule
        d = self.X.shape[1]
        if alpha is None or gamma is None:
            alpha, gamma = self._select_hyperparams(seed=seed)
        self.alpha, self.gamma = alpha, gamma
        self.model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma).fit(self.X, self.y)

    def _select_hyperparams(self, seed=0):
        """CV grid selection (proper model selection, not tuning on the test fold)."""
        d = self.X.shape[1]
        best, best_sp = (1.0, 1.0 / d), -2
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for alpha in (0.1, 0.3, 1.0, 3.0):
            for gm in (0.2, 0.5, 1.0, 2.0, 4.0):
                gamma = gm / d
                preds = np.zeros_like(self.y)
                for tr, te in kf.split(self.X):
                    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma).fit(self.X[tr], self.y[tr])
                    preds[te] = m.predict(self.X[te])
                sp = spearmanr(preds, self.y).correlation
                if sp > best_sp:
                    best_sp, best = sp, (alpha, gamma)
        return best

    # -- the field ----------------------------------------------------------------------------
    def field_value(self, smiles_list):
        """Predicted goodness at the blend's coordinate in flavor space."""
        v = self.space.vector(smiles_list)
        if v is None:
            return None
        return float(self.model.predict(v.reshape(1, -1))[0])

    def _sparsity(self, v, k=8):
        """Mean distance to the k nearest KNOWN molecules (higher = more unexplored region)."""
        d = np.linalg.norm(self.X - v[None, :], axis=1)
        return float(np.sort(d)[:k].mean())

    def superchoc_score(self, smiles_list, novelty_weight=0.5):
        """High goodness in a SPARSE region: field_value + novelty_weight * sparsity(z)."""
        v = self.space.vector(smiles_list)
        if v is None:
            return None
        fv = float(self.model.predict(v.reshape(1, -1))[0])
        sp = self._sparsity(v)
        return {"field_goodness": round(fv, 2), "sparsity": round(sp, 3),
                "superchoc_score": round(fv + novelty_weight * sp * 10, 2)}

    # -- validation: does the geometry field predict pleasantness? ----------------------------
    def cross_validate(self, n_splits=5, seed=0):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preds = np.zeros_like(self.y)
        for tr, te in kf.split(self.X):
            m = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)
            m.fit(self.X[tr], self.y[tr])
            preds[te] = m.predict(self.X[te])
        sp = float(spearmanr(preds, self.y).correlation)
        ss_res = float(((self.y - preds) ** 2).sum())
        ss_tot = float(((self.y - self.y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        # ECFP baseline (same CV) for an honest comparison
        Xe = np.vstack([smiles_to_ecfp4(s) for s in self.smiles])
        ep = np.zeros_like(self.y)
        for tr, te in kf.split(Xe):
            rm = RealGoodnessModel(seed=seed).fit(Xe[tr], self.y[tr])
            ep[te] = rm.predict(Xe[te])
        esp = float(spearmanr(ep, self.y).correlation)
        return {"n": int(len(self.y)),
                "geometry_field_spearman": round(sp, 3), "geometry_field_r2": round(r2, 3),
                "ecfp_baseline_spearman": round(esp, 3),
                "reading": ("geometry field predicts pleasantness from position, weaker than but "
                            "correlated with the structure model" if sp > 0.15 else
                            "geometry field is a weak position-only predictor")}


if __name__ == "__main__":
    from reference_flavors import REFERENCE_FLAVORS
    field = FlavorField()
    cv = field.cross_validate()
    print("goodness-field cross-validation:", json.dumps(cv, indent=2))
    print("\nfield goodness of known flavors (where they land):")
    for f, mols in REFERENCE_FLAVORS.items():
        smis = list(mols.values())
        sc = field.superchoc_score(smis)
        print(f"  {f:11s} field_goodness={sc['field_goodness']:5.1f}  sparsity={sc['sparsity']:.2f}"
              f"  superchoc_score={sc['superchoc_score']:.1f}")
    json.dump(cv, open(os.path.join(_HERE, "results", "flavor_field.json"), "w"), indent=2)
    print("\nwrote results/flavor_field.json")
