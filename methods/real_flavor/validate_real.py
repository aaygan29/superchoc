"""Pre-registered validation for the real-molecule flavor pipeline.

Checks (fixed before the run):
  1. Real goodness model: 5-fold CV Spearman >= 0.40 predicting real panel pleasantness
     (pleasantness is moderately predictable from structure; a real, honest bar).
  2. Learned mixture model beats the mean-pool baseline on the non-additive oracle by
     >= 0.10 Spearman (captures interaction that mean-pooling cannot).
  3. Safety invariant: every molecule in the generated suite passes the food-safety screen
     (verdict OK). A hard safety gate; any failure fails validation.
  4. Suite is novel: mean novelty-from-cocoa of suite candidates >= 0.5.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from real_data import build_keller_pleasantness
from featurize import featurize
from goodness_real import cross_validate
import mixture_model as mm
import suite_real as sr
from safety import screen


def run(verbose=True):
    # 1. goodness CV
    df = build_keller_pleasantness()
    X, mask = featurize(df["smiles"].tolist())
    y = df["pleasantness"].values[mask]
    cv = cross_validate(X, y, n_splits=5, seed=0)

    # 2. learned mixture vs mean-pool
    rows = [mm.compare(seed=s) for s in range(5)]
    mix_gain = float(np.mean([r["set_minus_meanpool"] for r in rows]))
    set_sp = float(np.mean([r["learned_set_model"] for r in rows]))
    mp_sp = float(np.mean([r["mean_pool_baseline"] for r in rows]))

    # 3 + 4. suite safety + novelty
    suite = sr.compose_suite(seed=0)
    all_safe = all(all(v == "OK" for v in c["member_safety"]) for c in suite["candidates"])
    # independent re-screen of every member SMILES (defense in depth)
    rescreen_ok = all(screen(s)["verdict"] == "OK"
                      for c in suite["candidates"] for s in c["smiles"])
    mean_novelty = float(np.mean([c["novelty_from_cocoa"] for c in suite["candidates"]]))

    results = {
        "goodness_cv_spearman": cv["spearman"],
        "goodness_cv_r2": cv["r2"],
        "goodness_n": cv["n"],
        "mixture_meanpool_spearman": round(mp_sp, 4),
        "mixture_learned_spearman": round(set_sp, 4),
        "mixture_gain": round(mix_gain, 4),
        "suite_n_candidates": len(suite["candidates"]),
        "suite_all_members_safe": bool(all_safe and rescreen_ok),
        "suite_mean_novelty_from_cocoa": round(mean_novelty, 4),
    }
    checks = {
        "goodness_ok": bool(cv["spearman"] >= 0.40),
        "mixture_ok": bool(mix_gain >= 0.10),
        "safety_gate": bool(all_safe and rescreen_ok),
        "novelty_ok": bool(mean_novelty >= 0.5),
    }
    results["checks"] = checks
    results["validated"] = bool(all(checks.values()))
    if verbose:
        print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    sys.exit(0 if run()["validated"] else 1)
