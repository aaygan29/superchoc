"""Produce a full, in-silico-validated recipe with a STRONG taste profile.

This ties the whole stack together into one deliverable: search molecule combinations for one
that (a) the recipe-level good-flavor classifier rates highly (P(good flavor)), and (b) has a
STRONG, focused taste/flavor profile (high profile_strength, not a muddy olfactory-white), then
optimize its component ratios against the classifier and characterize it on the five basic
tastes and the chef-descriptor flavor regions.

"Validated" here means: every scoring model it leans on is externally grounded or validated
(recipe classifier leave-one-flavor-out AUC 0.69; Snitz/Ravia mixture-distance; real Keller
pleasantness), and the safety screen gates it. It does NOT mean a human has tasted it; that
remains the wet-lab handoff (HANDOFF.md).

Search pool: known-safe flavor molecules (real, so the profile is meaningful and the safety
screen is informative), optionally extended with de-novo lego molecules. Objective:
    J = P(good flavor)  +  w_strength * profile_strength
subject to a hard safety gate. Ratios are then optimized to maximize P(good flavor) over the
simplex (a non-additive response, so the optimum is a real blend).
"""

from __future__ import annotations

import json
import os

import numpy as np

from real_data import build_keller_pleasantness
from safety import screen
import mixtures as mx
import recipe_classifier as rc
import taste_profile as tp
import composition as comp
import flavor_profile as fp

_HERE = os.path.dirname(__file__)


def design(seed=0, pool_size=90, n_components=6, n_search=400, w_strength=0.5,
           target_taste=None):
    rng = np.random.default_rng(seed)
    df = build_keller_pleasantness().reset_index(drop=True)
    # known, SAFE molecules only (real + safety-gated -> a meaningful, characterizable profile)
    pool = [(r["name"], r["smiles"]) for _, r in df.iterrows()
            if screen(r["smiles"])["verdict"] == "OK"]
    # keep the more pleasant half of the pool to bias toward good building blocks
    pl = {r["smiles"]: float(r["pleasantness"]) for _, r in df.iterrows()}
    pool.sort(key=lambda ns: pl.get(ns[1], 0), reverse=True)
    pool = pool[:pool_size]
    names = {s: n for n, s in pool}
    smis_pool = [s for _, s in pool]

    clf = rc.RecipeClassifier().fit()
    profiler = tp.TasteProfiler()

    def objective(smis):
        p = clf.score(smis)
        strg = profiler.profile_strength(smis) or 0.0
        j = (p or 0.0) + w_strength * strg
        if target_taste:  # optional: bias toward a target basic taste
            bt = profiler.basic_tastes(smis) or {}
            j += 0.5 * bt.get(target_taste, 0.0)
        return j, p, strg

    # random-restart hill climb
    best = None
    for _ in range(6):
        combo = list(rng.choice(smis_pool, size=n_components, replace=False))
        jb = objective(combo)[0]
        for _ in range(n_search // 6):
            cand = combo[:]
            cand[int(rng.integers(n_components))] = str(rng.choice(smis_pool))
            if len(set(cand)) < n_components:
                continue
            j = objective(cand)[0]
            if j > jb:
                combo, jb = cand, j
        if best is None or jb > best[1]:
            best = (combo, jb)

    combo = best[0]
    j, p, strg = objective(combo)

    # optimal ratios: maximize P(good flavor) over the simplex (non-additive response)
    opt = mx.optimize_ratios(combo, lambda w: float(clf.score(combo, weights=w)))

    # taste / flavor characterization
    cv = comp.composition_vector(combo)
    prof = profiler.profile(combo, comp_vec=cv)
    ref_mix = {f: list(m.values()) for f, m in __import__("reference_flavors").REFERENCE_FLAVORS.items()}
    space = mx.MixtureSpace([s for ms in ref_mix.values() for s in ms] + smis_pool)
    dists = {f: space.distance(combo, ms) for f, ms in ref_mix.items()}
    dists = {f: d for f, d in dists.items() if d is not None}

    # odor notes per component (nearest-molecule Leffingwell)
    desc_names, _ = fp.load_descriptor_table()
    components = []
    ratios = opt["best_proportions"]
    for i, s in enumerate(combo):
        components.append({
            "molecule": names.get(s, s), "smiles": s,
            "proportion": ratios[i] if i < len(ratios) else round(1.0 / len(combo), 3),
            "pleasantness": round(pl.get(s, float("nan")), 1),
            "safety": screen(s)["verdict"],
            "classes": sorted(comp.classify(s)),
        })

    package = {
        "n_components": len(combo),
        "good_flavor_probability": round(float(p), 3),
        "profile_strength": round(float(strg), 3),
        "objective": round(float(j), 3),
        "taste_profile": {
            "basic_tastes": prof["basic_tastes"],
            "dominant_taste": prof["dominant_taste"],
            "top_descriptors": prof["top_descriptors"],
            "flavor_region": prof["flavor_region"],
            "note": prof["note"],
        },
        "optimal_levels": {"proportions": ratios, "interior_optimum": opt["interior_optimum"],
                           "design_r2": opt["design_r2"], "reading": opt["reading"]},
        "perceptual_distance_to_known_flavors": {f: round(d, 3) for f, d in dists.items()},
        "nearest_known_flavor": min(dists, key=dists.get) if dists else None,
        "components": components,
        "how_to_make": ("Combine the components at the listed proportions (by perceptual weight) "
                        "in a neutral carrier; confirm FEMA/GRAS status and dose limits per "
                        "HANDOFF.md before any tasting."),
        "validation_note": ("All scores from externally-grounded/validated models (classifier "
                            "LOFO AUC 0.69; Snitz/Ravia mixture-distance; Keller pleasantness) + "
                            "safety gate. Not human-tasted; a hypothesis for a chemist/chef."),
    }
    return package


if __name__ == "__main__":
    pkg = design()
    print(f"validated recipe: P(good)={pkg['good_flavor_probability']} "
          f"strength={pkg['profile_strength']} dominant={pkg['taste_profile']['dominant_taste']}")
    print("basic tastes:", pkg["taste_profile"]["basic_tastes"])
    print("top notes:", pkg["taste_profile"]["top_descriptors"])
    print("flavor region:", pkg["taste_profile"]["flavor_region"])
    print("nearest known flavor:", pkg["nearest_known_flavor"],
          "| interior ratios:", pkg["optimal_levels"]["interior_optimum"])
    for c in pkg["components"]:
        print(f"  {c['proportion']:.3f}  {c['molecule']} ({'/'.join(c['classes'][:3])}) pl={c['pleasantness']}")
    json.dump(pkg, open(os.path.join(_HERE, "results", "validated_recipe.json"), "w"), indent=2)
    print("wrote results/validated_recipe.json")
