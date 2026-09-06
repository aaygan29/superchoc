"""FlavorDesigner: one system that unifies every analysis into a single objective.

Each thing we built becomes a component feeding one optimizer:

  chemical structure   -> ECFP4 features                       (featurize.py)
  taste / hedonics     -> pleasantness model, trained on a real (goodness_real.py)
                          human panel (Keller 2016); pleasantness is a primary
                          neuropsychological dimension of odor (Barnum & Hong 2022)
  flavor profiling     -> odor descriptors (Leffingwell)        (flavor_profile.py)
  toxicity             -> food-appropriate safety screen        (safety.py)
  mixture effects      -> learned mixture model (documented)    (mixture_model.py)
  novelty              -> distance from known flavors           (reference_flavors.py)

It proposes a flavor combination and instructions to make it, by MAXIMIZING predicted
pleasantness and novelty while MINIMIZING toxicity, in one scalar objective:

    J(recipe) = w_taste * (mean_pleasantness / 100)
              + w_novel * perceptual_novelty
              - w_tox   * toxicity_penalty

Toxicity is handled two ways: TOXIC molecules are hard-excluded, and CAUTION molecules are
allowed but penalized, so the optimizer actively minimizes toxicity rather than only
filtering it. The result is a single proposed recipe plus a make-it card.
"""

from __future__ import annotations

import json

import numpy as np

from recipes import _prep, _proportions, _novelty, _tanimoto  # noqa: reuse the built parts
import flavor_profile as fp


def design(seed=0, n_components=12, n_proposals=5, iters=600,
           w_taste=1.0, w_novel=0.6, w_tox=0.5,
           chem_novelty_min=0.5, perc_novelty_min=0.3,
           batch_g=10.0, aroma_load_pct=10.0):
    rows, X, model, ref_fp, ref_prof, desc_names = _prep()
    # eligibility: exclude TOXIC (hard); allow OK + CAUTION (CAUTION penalized in objective)
    elig = np.array([i for i, r in enumerate(rows) if r["safety"] in ("OK", "CAUTION")])
    rng = np.random.default_rng(seed)

    def tox_penalty(combo):
        return float(np.mean([1.0 if rows[i]["safety"] == "CAUTION" else 0.0 for i in combo]))

    def objective(combo):
        pl = float(model.predict(X[list(combo)]).mean()) / 100.0
        nv = _novelty(combo, rows, X, ref_fp, ref_prof)
        novelty = nv["perceptual_distance_to_nearest"]
        tox = tox_penalty(combo)
        J = w_taste * pl + w_novel * novelty - w_tox * tox
        return J, {"mean_pleasantness": round(pl * 100, 2), "perceptual_novelty": novelty,
                   "chem_novelty": nv["chem_distance_to_nearest"], "toxicity_penalty": round(tox, 3),
                   "objective": round(J, 4), "_nv": nv}

    def feasible(combo):
        nv = _novelty(combo, rows, X, ref_fp, ref_prof)
        return (nv["chem_distance_to_nearest"] >= chem_novelty_min
                and nv["perceptual_distance_to_nearest"] >= perc_novelty_min)

    proposals = []
    attempts = 0
    while len(proposals) < n_proposals and attempts < n_proposals * 40:
        attempts += 1
        combo = tuple(rng.choice(elig, size=n_components, replace=False))
        t = 0
        while not feasible(combo) and t < 150:
            combo = tuple(rng.choice(elig, size=n_components, replace=False)); t += 1
        if not feasible(combo):
            continue
        bestJ, _ = objective(combo)
        for _ in range(iters):
            cand = list(combo); cand[rng.integers(n_components)] = int(rng.choice(elig))
            if len(set(cand)) < n_components or not feasible(cand):
                continue
            cand = tuple(cand)
            J, _ = objective(cand)
            if J > bestJ:
                combo, bestJ = cand, J
        if frozenset(combo) not in {frozenset(c) for c, _ in proposals}:
            proposals.append((combo, bestJ))

    proposals.sort(key=lambda t: t[1], reverse=True)
    carrier_g = batch_g * (1 - aroma_load_pct / 100.0)
    aroma_g = batch_g * (aroma_load_pct / 100.0)

    out = []
    for rank, (combo, J) in enumerate(proposals, 1):
        _, comp_metrics = objective(combo)
        nv = comp_metrics.pop("_nv")
        prof_top = fp.top_descriptors(nv["_profile"] if "_profile" in nv else
                                      _novelty(combo, rows, X, ref_fp, ref_prof)["_profile"],
                                      desc_names)
        pct = _proportions([model.predict(X[i:i+1])[0] for i in combo])
        comps = []
        for i, p in zip(combo, pct):
            dv = rows[i]["descriptors"]
            notes = [desc_names[j] for j in np.argsort(dv)[::-1][:3] if dv[j] > 0] if dv is not None else []
            comps.append({"molecule": rows[i]["name"], "smiles": rows[i]["smiles"],
                          "cas": rows[i]["cas"], "percent_of_aroma_w_w": round(float(p), 2),
                          "mass_mg_in_batch": round(float(p) / 100.0 * aroma_g * 1000.0, 1),
                          "odor_notes": notes, "safety": rows[i]["safety"]})
        comps.sort(key=lambda c: c["percent_of_aroma_w_w"], reverse=True)
        name = "-".join(d for d, _ in prof_top[:3]) + " accord" if prof_top else "novel accord"
        out.append({"rank": rank, "flavor_name": name,
                    "objective_score": comp_metrics["objective"],
                    "objective_breakdown": {k: comp_metrics[k] for k in
                        ("mean_pleasantness", "perceptual_novelty", "chem_novelty", "toxicity_penalty")},
                    "flavor_profile_top_notes": prof_top, "n_components": len(combo),
                    "batch": {"total_g": batch_g, "aroma_g": round(aroma_g, 3),
                              "carrier_g": round(carrier_g, 3),
                              "carrier": "propylene glycol (food grade)"},
                    "components": comps})

    return {
        "system": "FlavorDesigner (unified: chemical + hedonic/taste + odor-profile + toxicity)",
        "objective": f"maximize {w_taste}*taste + {w_novel}*novelty - {w_tox}*toxicity",
        "note": ("One model over real inputs: pleasantness from a human panel (Keller 2016), "
                 "odor profile from Leffingwell, chemistry from ECFP4, toxicity from a food-"
                 "appropriate screen. TOXIC molecules are excluded; CAUTION molecules are "
                 "penalized so toxicity is actively minimized. Proportions are heuristic "
                 "starting points. NOT a safety clearance; see HANDOFF.md."),
        "proposals": out,
    }


def to_markdown(d):
    L = [f"# {d['system']}", "", f"**Objective:** {d['objective']}", "", f"> {d['note']}", ""]
    for p in d["proposals"]:
        b = p["objective_breakdown"]
        prof = ", ".join(f"{x} ({v})" for x, v in p["flavor_profile_top_notes"]) or "n/a"
        L += [f"## Proposal {p['rank']}: \"{p['flavor_name']}\"  (objective {p['objective_score']})",
              "",
              f"Taste (pleasantness): {b['mean_pleasantness']}/100 | perceptual novelty "
              f"{b['perceptual_novelty']} | chemical novelty {b['chem_novelty']} | toxicity "
              f"penalty {b['toxicity_penalty']}", "",
              f"Flavor profile: {prof}", "",
              f"Batch: {p['batch']['total_g']} g = {p['batch']['aroma_g']} g aroma in "
              f"{p['batch']['carrier_g']} g {p['batch']['carrier']}.", "",
              "| molecule | % aroma | mg/batch | odor notes | CAS | safety |",
              "|---|---|---|---|---|---|"]
        for c in p["components"]:
            L.append(f"| {c['molecule']} | {c['percent_of_aroma_w_w']} | {c['mass_mg_in_batch']} | "
                     f"{', '.join(c['odor_notes']) or 'n/a'} | {c['cas'] or 'n/a'} | {c['safety']} |")
        L.append("")
    L += ["## How to make", "Weigh each component (mg) into an amber vial (fume hood), add "
          "food-grade propylene glycol to the carrier mass, mix, equilibrate 24-48 h, evaluate "
          "by smell first. Do NOT taste without GRAS/FEMA confirmation (HANDOFF.md)."]
    return "\n".join(L)


if __name__ == "__main__":
    d = design()
    print(to_markdown(d))
