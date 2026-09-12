"""Recipe-level "good flavor" classifier: learn what a good flavor COMBO looks like, then use
it to score recipes and drive optimal-ratio search.

The per-molecule goodness model (goodness_real.py) predicts pleasantness of ONE molecule. This
module learns at the RECIPE level: it is trained to tell a real, good flavor combination
(chocolate, vanilla, coffee, banana, strawberry, rose, plus single high-pleasantness molecules
like vanillin) apart from an arbitrary random blend. That gives a single non-additive "good
flavor" response over a mixture, which is exactly what was missing to make ratio optimization
non-degenerate: because the features depend on the component PROPORTIONS, optimizing the
classifier score over the simplex yields a real interior blend, not a single molecule.

Data
  Positives: the six reference flavors and random SUBSETS of each (a subset of a good flavor's
    key odorants is still flavor-like), plus top-quartile-pleasantness single Keller molecules.
  Negatives: random blends of Keller molecules (size-matched) and bottom-quartile singles.
Features (mixture level, proportion-aware)
  Snitz physicochemical mixture vector (mixtures.MixtureSpace) + composition class fractions +
  n_components + olfactory-white risk + goodness-model pleasantness (mean/max/min over parts).
Models
  RandomForest + GradientBoosting + LogisticRegression, plus a soft-vote ensemble.
Validation (honest)
  LEAVE-ONE-FLAVOR-OUT: train on five flavors (+ singles + randoms), test on the held-out
  flavor's positives vs random negatives. This tests whether "good-flavor-ness" generalizes to
  an UNSEEN flavor, so subsets cannot leak. We report per-model and ensemble ROC-AUC.
  Caveat: positives are "known real flavors / high-pleasantness", negatives are "random blends".
  The classifier therefore learns flavor-like COHERENCE, not proven deliciousness. There is no
  human combo-deliciousness ground truth; a positive result means "looks like a real flavor",
  which is a useful prior for ranking recipes, not a guarantee they taste good.

Refs: reference-flavor odorants are literature key-odorant sets; features follow Snitz 2013
(mixture physicochemical distance) and the repo's composition/goodness models.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from goodness_real import RealGoodnessModel
import composition as comp
import mixtures as mx
from reference_flavors import REFERENCE_FLAVORS

_HERE = os.path.dirname(__file__)
_CLASSES = ['alcohol', 'aldehyde', 'alkene', 'amine', 'aromatic', 'carboxylic_acid', 'ester',
            'ether', 'furan', 'ketone', 'lactone', 'phenol', 'pyrazine', 'sulfur',
            'terpenoid_like']


def _context():
    df = build_keller_pleasantness().reset_index(drop=True)
    smis, X, y = [], [], []
    for _, r in df.iterrows():
        try:
            fp = smiles_to_ecfp4(r["smiles"])
        except Exception:
            continue
        smis.append(r["smiles"]); X.append(fp); y.append(float(r["pleasantness"]))
    X = np.vstack(X); y = np.array(y)
    model = RealGoodnessModel(seed=0).fit(X, y)
    ref_smiles = [s for ms in REFERENCE_FLAVORS.values() for s in ms.values()]
    space = mx.MixtureSpace(ref_smiles + smis)
    return {"keller_smiles": smis, "keller_y": y, "model": model, "space": space}


def featurize_recipe(smiles_list, ctx, weights=None):
    """Mixture-level feature vector for one recipe (proportion-aware via weights)."""
    space = ctx["space"]
    vec = space.vector(smiles_list, weights=weights)
    if vec is None:
        return None
    cv = comp.composition_vector(smiles_list)
    comp_vec = [float(cv.get(c, 0.0)) for c in _CLASSES]
    try:
        pls = ctx["model"].predict(np.vstack([smiles_to_ecfp4(s) for s in smiles_list]))
        pl_stats = [float(pls.mean()), float(pls.max()), float(pls.min())]
    except Exception:
        pl_stats = [0.0, 0.0, 0.0]
    white = mx.olfactory_white_risk(len(smiles_list))["white_risk"]
    return np.concatenate([vec, comp_vec, [len(smiles_list), white], pl_stats])


def build_dataset(ctx, seed=0, subsets_per_flavor=40):
    rng = np.random.default_rng(seed)
    Xr, y, groups = [], [], []

    # positives: reference flavors + random subsets (grouped by flavor for leave-flavor-out CV)
    for flavor, mols in REFERENCE_FLAVORS.items():
        smis = list(mols.values())
        recipes = [smis]
        for _ in range(subsets_per_flavor):
            k = int(rng.integers(2, len(smis) + 1))
            recipes.append(list(rng.choice(smis, size=k, replace=False)))
        for rec in recipes:
            f = featurize_recipe(rec, ctx)
            if f is not None:
                Xr.append(f); y.append(1); groups.append(flavor)

    # positives: top-quartile single Keller molecules (good 1-component "recipes")
    ks, ky = ctx["keller_smiles"], ctx["keller_y"]
    hi = np.quantile(ky, 0.75); lo = np.quantile(ky, 0.25)
    for s, yy in zip(ks, ky):
        if yy >= hi:
            f = featurize_recipe([s], ctx)
            if f is not None:
                Xr.append(f); y.append(1); groups.append("single_good")

    # negatives: random blends (size-matched to flavor recipes) + bottom-quartile singles
    sizes = [2, 3, 4, 5, 6, 7, 8]
    for _ in range(300):
        k = int(rng.choice(sizes))
        rec = list(rng.choice(ks, size=k, replace=False))
        f = featurize_recipe(rec, ctx)
        if f is not None:
            Xr.append(f); y.append(0); groups.append("random_blend")
    for s, yy in zip(ks, ky):
        if yy <= lo:
            f = featurize_recipe([s], ctx)
            if f is not None:
                Xr.append(f); y.append(0); groups.append("single_bad")

    return np.vstack(Xr), np.array(y), np.array(groups)


def _models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "gradient_boost": GradientBoostingClassifier(random_state=0),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def validate_leave_flavor_out(ctx, seed=0):
    """Train on five flavors (+ singles + randoms), test on the held-out flavor vs randoms."""
    X, y, groups = build_dataset(ctx, seed=seed)
    flavors = list(REFERENCE_FLAVORS.keys())
    per_model = {m: [] for m in list(_models()) + ["ensemble"]}
    for held in flavors:
        # test = held-out flavor positives + all random-blend negatives
        test_mask = (groups == held) | (groups == "random_blend")
        train_mask = ~((groups == held))  # drop held flavor from training; keep other negatives
        # ensure test negatives are not in train
        train_mask = train_mask & ~((groups == "random_blend") & test_mask)
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        if len(np.unique(yte)) < 2:
            continue
        probs = []
        for name, clf in _models().items():
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            per_model[name].append(roc_auc_score(yte, p))
            probs.append(p)
        per_model["ensemble"].append(roc_auc_score(yte, np.mean(probs, axis=0)))
    return {m: round(float(np.mean(v)), 3) for m, v in per_model.items() if v}


class RecipeClassifier:
    """Deployment classifier: ensemble of RF + GBM + Logistic, trained on all data."""

    def __init__(self, ctx=None):
        self.ctx = ctx or _context()
        self.models = None

    def fit(self, seed=0):
        X, y, _ = build_dataset(self.ctx, seed=seed)
        self.models = _models()
        for clf in self.models.values():
            clf.fit(X, y)
        return self

    def score(self, smiles_list, weights=None):
        """Ensemble P(good flavor) for a recipe; proportion-aware via weights."""
        f = featurize_recipe(smiles_list, self.ctx, weights=weights)
        if f is None:
            return None
        ps = [clf.predict_proba(f.reshape(1, -1))[0, 1] for clf in self.models.values()]
        return float(np.mean(ps))


if __name__ == "__main__":
    ctx = _context()
    aucs = validate_leave_flavor_out(ctx)
    print("leave-one-flavor-out ROC-AUC:", json.dumps(aucs, indent=2))
    clf = RecipeClassifier(ctx).fit()
    demos = {
        "vanilla (known good)": list(REFERENCE_FLAVORS["vanilla"].values()),
        "chocolate (known good)": list(REFERENCE_FLAVORS["chocolate"].values()),
        "vanillin+peppermint(menthol)": ["O=Cc1ccc(O)c(OC)c1", "CC(C)C1CCC(C)CC1O"],
        "random-ish blend": ["CCCCCCCCCC(=O)O", "SCc1ccco1", "CC(C)(C)c1ccccc1"],
    }
    print("demo scores:")
    for name, rec in demos.items():
        print(f"  {clf.score(rec):.3f}  {name}")
    out = {"leave_one_flavor_out_auc": aucs,
           "demo_scores": {k: round(clf.score(v), 3) for k, v in demos.items()}}
    json.dump(out, open(os.path.join(_HERE, "results", "recipe_classifier.json"), "w"), indent=2)
    print("wrote results/recipe_classifier.json")
