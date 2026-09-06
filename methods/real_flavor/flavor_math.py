"""Flavor-space mathematics: a tastiness functional, its provable properties, and their
empirical validation.

HONEST SCOPE. You cannot mathematically *prove* that a physical mixture tastes good;
tastiness is an empirical property of human perception. What can be made rigorous:
  (A) MATHEMATICAL properties of the tastiness/novelty FUNCTIONALS we optimize (bounds,
      mixture bounds, metric axioms). These are theorems (see FlavorMath.lean).
  (B) EMPIRICAL validity of the metric: does the tastiness score predict held-out human
      pleasantness, is it calibrated, is it stable? Tested against Keller 2016.
This module implements the functionals and the executable property tests for (A), and the
empirical tests for (B). It then applies the validated metric to the new flavor combos.

Definitions.
  Let a molecule i have model-predicted pleasantness p_i in [0,100].
  A recipe R is a set of molecules with weights w_i >= 0, sum w_i = 1.
  Tastiness functional (additive estimate):   T(R) = sum_i w_i * p_i.
  Novelty (perceptual):  N(R) = min over known flavors F of cosine_distance(profile(R), profile(F)).
  Objective:             J(R) = a*T(R)/100 + b*N(R) - c*tox(R),  a,b,c >= 0.

Provable properties (theorems, mirrored in Lean):
  P1 (bounded taste):  min_i p_i <= T(R) <= max_i p_i, and T(R) in [0,100].
  P2 (no free lunch):  T(R) <= max_i p_i  (a blend is never tastier than its best part
                       UNDER the additive estimate; this is exactly why a learned,
                       non-additive mixture model is needed to exceed it -- and it is the
                       formal statement of the user's question "can a combo beat its parts?").
  P3 (novelty is a pseudometric component): cosine_distance in [0,2], symmetric, d(x,x)=0.
  P4 (objective bound): 0 - c <= J(R) <= a + b*2  for weights/inputs in range.
"""

from __future__ import annotations

import numpy as np


def tastiness(pleasantness, weights=None):
    p = np.asarray(pleasantness, dtype=float)
    if weights is None:
        weights = np.ones_like(p) / len(p)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float((w * p).sum())


def cosine_distance(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(a @ b / (na * nb))


def objective(T, N, tox, a=1.0, b=0.6, c=0.5):
    return a * (T / 100.0) + b * N - c * tox


# ---- executable checks of the theorems (property-based, random inputs) ----
def check_properties(n_trials=5000, seed=0):
    rng = np.random.default_rng(seed)
    results = {"P1_bounded_taste": True, "P2_no_free_lunch": True,
               "P3_cosine_metric": True, "P4_objective_bound": True}
    for _ in range(n_trials):
        k = rng.integers(2, 15)
        p = rng.uniform(0, 100, size=k)
        w = rng.uniform(0, 1, size=k)
        T = tastiness(p, w)
        if not (p.min() - 1e-9 <= T <= p.max() + 1e-9 and 0 <= T <= 100):
            results["P1_bounded_taste"] = False
        if T > p.max() + 1e-9:
            results["P2_no_free_lunch"] = False
        d = rng.integers(2, 20)
        x = rng.normal(size=d); y = rng.normal(size=d)
        dxy = cosine_distance(x, y)
        if not (-1e-9 <= dxy <= 2 + 1e-9 and abs(cosine_distance(x, y) - cosine_distance(y, x)) < 1e-9
                and abs(cosine_distance(x, x)) < 1e-6):
            results["P3_cosine_metric"] = False
        N = rng.uniform(0, 2); tox = rng.uniform(0, 1)
        J = objective(T, N, tox)
        if not (-0.5 - 1e-9 <= J <= 1.0 + 0.6 * 2 + 1e-9):
            results["P4_objective_bound"] = False
    results["all_passed"] = all(results.values())
    return results


# ---- empirical validity of the metric against real human data ----
def empirical_validity(seed=0):
    """Does the tastiness metric (model pleasantness) predict held-out HUMAN pleasantness,
    and is P2 (blend <= best part, additive) something real data motivates moving beyond?"""
    from scipy.stats import spearmanr
    from real_data import build_keller_pleasantness
    from featurize import featurize
    from goodness_real import cross_validate
    df = build_keller_pleasantness()
    X, mask = featurize(df["smiles"].tolist())
    y = df["pleasantness"].values[mask]
    cv = cross_validate(X, y, n_splits=5, seed=seed)
    return {"metric_vs_human_spearman": cv["spearman"], "metric_vs_human_r2": cv["r2"],
            "n": cv["n"]}


if __name__ == "__main__":
    import json
    props = check_properties()
    print("PROPERTY CHECKS:", json.dumps(props))
    print("EMPIRICAL VALIDITY:", json.dumps(empirical_validity()))
