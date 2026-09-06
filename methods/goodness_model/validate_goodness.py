"""Validate the goodness model + combo generator against the oracle.

The central honesty question (the Goodhart control): a generator that maximizes the
MODEL's score is only useful if the combos it proposes are ALSO good on the TRUE oracle.
If model-predicted goodness is high but true goodness is not, the generator is just
exploiting model error, not discovering real flavor. We test both.

Pre-registered success criteria (fixed before the final run), averaged over seeds:
  1. Goodness model held-out R2 >= 0.50 (embedding -> goodness is learnable).
  2. Generated combos' mean TRUE goodness > mean TRUE goodness of random combos + 0.02.
  3. Generated combos reach the chocolate reference: mean TRUE goodness >= ref value.
  4. Goodhart control (two parts): the model must rank a POPULATION of random combos
     correctly, Pearson(model_pred, true) over 200 random combos >= 0.6 (a well-posed test,
     unlike correlating over the handful of near-tied winners); AND on the generated combos
     the mean gap (pred - true) must be <= 0.10 (the model does not over-promise its picks).
  5. Novelty: mean cosine distance of generated combos from the chocolate combo >= 0.2
     (by construction, but verified).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from flavor_space import FlavorSpace
from goodness_model import evaluate_model
from compose import compose_combos, random_combos


def run(n_seeds=8, k=4, verbose=True):
    r2s, gen_true, rand_true, ref_vals = [], [], [], []
    goodhart_corr, goodhart_gap, novelties = [], [], []

    for seed in range(n_seeds):
        space = FlavorSpace(seed=seed)
        model, r2 = evaluate_model(space, n_train=400, seed=seed)
        r2s.append(r2)

        ref = space.chocolate_reference(k=k)
        ref_vals.append(ref["value"])

        gen = compose_combos(space, model, k=k, n_combos=5, novelty_min=0.2,
                             ref_embedding=ref["embedding"], iters=300, seed=seed)
        rnd = random_combos(space, k=k, n_combos=200, seed=seed)

        gt = np.array([g["true_goodness"] for g in gen])
        pt = np.array([g["predicted_goodness"] for g in gen])
        rt = np.array([r["true_goodness"] for r in rnd])
        gen_true.append(gt.mean())
        rand_true.append(rt.mean())
        novelties.append(np.mean([g["novelty_from_ref"] for g in gen]))

        # Goodhart control part 1: does the model rank a VARIED population correctly?
        rand_true_vec = rt
        rand_pred_vec = np.array([model.predict(space.mixture_embedding(r["combo"]))[0]
                                  for r in rnd])
        if np.std(rand_pred_vec) > 1e-9 and np.std(rand_true_vec) > 1e-9:
            goodhart_corr.append(float(np.corrcoef(rand_pred_vec, rand_true_vec)[0, 1]))
        # Goodhart control part 2: model does not over-promise on its own picks.
        goodhart_gap.append(float(np.mean(pt - gt)))

    r2s = np.array(r2s)
    gen_true = np.array(gen_true)
    rand_true = np.array(rand_true)
    ref_vals = np.array(ref_vals)
    corr = float(np.mean(goodhart_corr)) if goodhart_corr else float("nan")
    gap = float(np.mean(goodhart_gap))

    results = {
        "n_seeds": n_seeds, "k": k,
        "goodness_r2_mean": round(float(r2s.mean()), 4),
        "generated_true_mean": round(float(gen_true.mean()), 4),
        "random_true_mean": round(float(rand_true.mean()), 4),
        "gen_minus_random": round(float(gen_true.mean() - rand_true.mean()), 4),
        "chocolate_ref_mean": round(float(ref_vals.mean()), 4),
        "gen_minus_choc": round(float(gen_true.mean() - ref_vals.mean()), 4),
        "goodhart_pred_true_corr": round(corr, 4),
        "goodhart_pred_minus_true_gap": round(gap, 4),
        "novelty_from_choc_mean": round(float(np.mean(novelties)), 4),
    }
    checks = {
        "model_learnable": bool(r2s.mean() >= 0.50),
        "beats_random": bool((gen_true.mean() - rand_true.mean()) >= 0.02),
        "reaches_chocolate": bool(gen_true.mean() >= ref_vals.mean()),
        "not_goodharting": bool(corr == corr and corr >= 0.6 and gap <= 0.10),
        "novel": bool(np.mean(novelties) >= 0.2),
    }
    results["checks"] = checks
    results["validated"] = bool(all(checks.values()))

    if verbose:
        print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    sys.exit(0 if res["validated"] else 1)
