"""Significance layer: confidence intervals + permutation p-values + FDR correction for the
pipeline's headline results, so each number is an effect + CI + corrected p, not a bare point
estimate.

Methods (all nonparametric, no distributional assumptions):
  - bootstrap_ci: percentile bootstrap over the paired observations.
  - permutation_p: shuffle the labels/targets, recompute the statistic; p = fraction of
    permuted |stat| >= observed |stat| (two-sided).
  - benjamini_hochberg: FDR control across the family of tests.

Honest caveat recorded in the output: the mixture-similarity pairs (Snitz, Ravia) SHARE
component molecules, so the pairs are not fully independent. The label-permutation test here
answers "is there any distance<->similarity association beyond chance labeling", a first-order
significance check; a fully rigorous test would permute at the molecule level. We flag this
rather than overstate the p-value.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(__file__)


def _spearman(a, b):
    return float(spearmanr(a, b).correlation)


def bootstrap_ci(a, b, stat_fn=_spearman, n_boot=2000, seed=0, ci=95):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(stat_fn(a[idx], b[idx]))
    lo, hi = np.percentile(stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return round(float(lo), 3), round(float(hi), 3)


def permutation_p(a, b, stat_fn=_spearman, n_perm=5000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    obs = abs(stat_fn(a, b))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        ge += abs(stat_fn(a, rng.permutation(b))) >= obs
    return float((ge + 1) / (n_perm + 1))  # add-one (never reports p=0)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return [round(float(x), 4) for x in adj]


def _entry(name, a, b, effect_name="spearman", n_boot=2000, n_perm=5000, seed=0):
    eff = _spearman(a, b)
    ci = bootstrap_ci(a, b, n_boot=n_boot, seed=seed)
    p = permutation_p(a, b, n_perm=n_perm, seed=seed)
    return {"test": name, "n": int(len(a)), effect_name: round(eff, 3),
            "ci95": ci, "perm_p": round(p, 5)}


def run(n_boot=2000, n_perm=5000, seed=0):
    import mixtures as mx
    import flavor_field as ff

    entries = []

    # 1-2. Mixture perceptual-distance vs human similarity (external validations)
    ds, ss = mx.snitz_vectors()
    entries.append(_entry("mixture_distance_vs_similarity_snitz2013", ds, ss,
                          n_boot=n_boot, n_perm=n_perm, seed=seed))
    dr, sr = mx.ravia_vectors()
    entries.append(_entry("mixture_distance_vs_similarity_ravia2020", dr, sr,
                          n_boot=n_boot, n_perm=n_perm, seed=seed))

    # 3. Goodness field: cross-validated predicted pleasantness vs truth
    field = ff.FlavorField()
    from sklearn.model_selection import KFold
    from sklearn.kernel_ridge import KernelRidge
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(field.y)
    for tr, te in kf.split(field.X):
        m = KernelRidge(alpha=field.alpha, kernel="rbf", gamma=field.gamma).fit(field.X[tr], field.y[tr])
        preds[te] = m.predict(field.X[te])
    entries.append(_entry("goodness_field_cv_vs_pleasantness", preds, field.y,
                          n_boot=n_boot, n_perm=n_perm, seed=seed))

    # FDR across the family
    padj = benjamini_hochberg([e["perm_p"] for e in entries])
    for e, pa in zip(entries, padj):
        e["fdr_bh"] = pa
        e["significant_fdr_0.05"] = bool(pa < 0.05)

    out = {"tests": entries,
           "method": "percentile bootstrap CI + label-permutation p + Benjamini-Hochberg FDR",
           "caveat": ("Snitz/Ravia mixture pairs share component molecules (not fully "
                      "independent); the permutation test is a first-order check, not a "
                      "molecule-level exact test.")}
    return out


if __name__ == "__main__":
    out = run()
    print(f"{'test':44s} {'n':>4} {'effect':>7} {'ci95':>16} {'perm_p':>8} {'fdr':>7} sig")
    for e in out["tests"]:
        print(f"{e['test']:44s} {e['n']:>4} {e['spearman']:>7.3f} "
              f"{str(e['ci95']):>16} {e['perm_p']:>8.4f} {e['fdr_bh']:>7.4f} "
              f"{'*' if e['significant_fdr_0.05'] else ''}")
    json.dump(out, open(os.path.join(_HERE, "results", "significance.json"), "w"), indent=2)
    print("\nwrote results/significance.json")
