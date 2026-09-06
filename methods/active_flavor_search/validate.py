"""Validation experiment: does online active search beat random screening?

Honest question, pre-registered success bar. Both methods run on the SAME synthetic
ground-truth instance per seed and we compare best-found deliciousness for a fixed
tasting budget. See docs/GROUND_TRUTH.md for the evaluation contract.

Pre-registered success criteria (fixed before the final run):
  1. Mean final best-found: active >= random + 0.02 (absolute).
  2. Active wins (>= random) on at least 75% of seeds.
  3. Sample efficiency: active's mean best at HALF the budget >= random's mean best
     at the FULL budget. (Online search should get more from fewer tastings.)
If any fail, the method is reported NOT validated and is not committed as usable.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ground_truth import SyntheticFlavorSpace
from active_search import run_active_search, run_random_search


def run(n_seeds=25, budget=50, strategy="ucb", verbose=True):
    half = budget // 2
    active_final, random_final = [], []
    active_half = []
    wins = 0

    for seed in range(n_seeds):
        space_a = SyntheticFlavorSpace(seed=seed)
        space_r = SyntheticFlavorSpace(seed=seed)

        a = run_active_search(space_a, budget=budget, strategy=strategy, seed=seed)
        r = run_random_search(space_r, budget=budget, seed=seed)

        af, rf = a["best_so_far"][-1], r["best_so_far"][-1]
        active_final.append(af)
        random_final.append(rf)
        active_half.append(a["best_so_far"][half - 1])
        wins += int(af >= rf)

    active_final = np.array(active_final)
    random_final = np.array(random_final)
    active_half = np.array(active_half)

    win_rate = wins / n_seeds
    mean_gain = float(active_final.mean() - random_final.mean())
    efficiency_gap = float(active_half.mean() - random_final.mean())

    results = {
        "n_seeds": n_seeds,
        "budget": budget,
        "strategy": strategy,
        "active_mean_final": round(float(active_final.mean()), 4),
        "random_mean_final": round(float(random_final.mean()), 4),
        "mean_gain": round(mean_gain, 4),
        "active_win_rate": round(float(win_rate), 4),
        "active_mean_at_half_budget": round(float(active_half.mean()), 4),
        "efficiency_gap_half_vs_full": round(efficiency_gap, 4),
    }

    checks = {
        "gain_ok": mean_gain >= 0.02,
        "winrate_ok": win_rate >= 0.75,
        "efficiency_ok": efficiency_gap >= 0.0,
    }
    results["checks"] = checks
    results["validated"] = all(checks.values())

    if verbose:
        print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    sys.exit(0 if res["validated"] else 1)
