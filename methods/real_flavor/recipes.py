"""Generate flavor RECIPES: multi-component blends with proportions, batch formulation
cards ("how to make it"), real odor profiles, and combination-level novelty analysis.

What feeds the generator (all real data through the proper channels):
  - Real pleasantness labels (Keller & Vosshall 2016 human panel) -> goodness model.
  - Real odor descriptors (Leffingwell, 113 notes) -> per-molecule and per-recipe flavor
    profile, and the definition of a NOVEL flavor character.
  - Food-appropriate safety screen (safety.py) -> only OK molecules are eligible.
  - Reference natural/known flavors (reference_flavors.py) -> novelty is measured against
    them on BOTH a chemical axis (ECFP/Tanimoto) and a perceptual axis (descriptor cosine).

A recipe is a blend of ~12 key odorants. Formulated flavors are built from a set of key
compounds (recombination studies reproduce e.g. chocolate aroma from ~20-30); the thousands
of trace volatiles in a bean are mostly sub-threshold. Proportions are a documented
heuristic starting point, to be balanced by a flavorist with odor-threshold data.

NOVELTY, honestly: we require each recipe to be far from every reference flavor chemically
AND perceptually, and to share few components with any reference. This shows a flavor
character distinct from known/public flavors. It CANNOT prove absence from proprietary
trade-secret formulas, which are not publicly knowable.
"""

from __future__ import annotations

import json

import numpy as np

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from goodness_real import RealGoodnessModel
from safety import screen
from reference_flavors import REFERENCE_FLAVORS
import flavor_profile as fp


def _tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def _prep():
    from rdkit import Chem
    df = build_keller_pleasantness().reset_index(drop=True)
    desc_names, cid2desc = fp.load_descriptor_table()
    _, smi2desc = fp.canonical_smiles_index()

    rows, feats = [], []
    for _, r in df.iterrows():
        try:
            fpv = smiles_to_ecfp4(r["smiles"])
        except Exception:
            continue
        feats.append(fpv)
        rows.append({
            "name": r["name"], "smiles": r["smiles"], "cas": str(r.get("CAS", "")),
            "cid": int(r["CID"]), "pleasantness": float(r["pleasantness"]),
            "safety": screen(r["smiles"])["verdict"],
            "descriptors": cid2desc.get(int(r["CID"])),  # 113-vec or None
        })
    X = np.vstack(feats)
    model = RealGoodnessModel(seed=0).fit(X, np.array([r["pleasantness"] for r in rows]))

    # reference flavor fingerprints + descriptor profiles
    ref_fp, ref_prof = {}, {}
    for flav, mols in REFERENCE_FLAVORS.items():
        fps, vecs = [], []
        for s in mols.values():
            try:
                m = Chem.MolFromSmiles(s); cs = Chem.MolToSmiles(m)
                fps.append(smiles_to_ecfp4(s))
                if cs in smi2desc:
                    vecs.append(smi2desc[cs])
            except Exception:
                continue
        ref_fp[flav] = fps
        ref_prof[flav] = fp.recipe_profile(vecs)
    return rows, X, model, ref_fp, ref_prof, desc_names


def _proportions(pred, temperature=8.0):
    p = np.array(pred, dtype=float)
    w = np.exp((p - p.max()) / temperature); w = w / w.sum()
    return 100.0 * w


def _novelty(component_idx, rows, X, ref_fp, ref_prof):
    comp_fps = [X[i] for i in component_idx]
    comp_vecs = [rows[i]["descriptors"] for i in component_idx]
    prof = fp.recipe_profile(comp_vecs)
    # chemical distance to nearest reference flavor
    chem = {}
    for flav, fps in ref_fp.items():
        sims = [max(_tanimoto(c, r) for r in fps) for c in comp_fps]
        chem[flav] = 1.0 - float(np.mean(sims))
    # perceptual distance to nearest reference flavor
    perc = {flav: fp.cosine_distance(prof, ref_prof[flav]) for flav in ref_prof}
    all_ref = [r for fps in ref_fp.values() for r in fps]
    n_close = sum(1 for c in comp_fps if max(_tanimoto(c, r) for r in all_ref) > 0.6)
    nearest_chem = min(chem, key=chem.get)
    nearest_perc = min(perc, key=perc.get)
    return {"chem_distance_to_nearest": round(chem[nearest_chem], 3),
            "nearest_chem_flavor": nearest_chem,
            "perceptual_distance_to_nearest": round(perc[nearest_perc], 3),
            "nearest_perceptual_flavor": nearest_perc,
            "components_resembling_known": int(n_close),
            "_profile": prof}


def compose_recipes(seed=0, n_components=12, n_recipes=5, iters=500,
                    chem_novelty_min=0.55, perc_novelty_min=0.35, max_resembling=3,
                    batch_g=10.0, aroma_load_pct=10.0):
    rows, X, model, ref_fp, ref_prof, desc_names = _prep()
    elig = np.array([i for i, r in enumerate(rows) if r["safety"] == "OK"])
    rng = np.random.default_rng(seed)

    def score(combo):
        return float(model.predict(X[list(combo)]).mean())

    def ok(combo):
        nv = _novelty(combo, rows, X, ref_fp, ref_prof)
        good = (nv["chem_distance_to_nearest"] >= chem_novelty_min
                and nv["perceptual_distance_to_nearest"] >= perc_novelty_min
                and nv["components_resembling_known"] <= max_resembling)
        return good, nv

    made = []
    attempts = 0
    while len(made) < n_recipes and attempts < n_recipes * 40:
        attempts += 1
        combo = tuple(rng.choice(elig, size=n_components, replace=False))
        good, _ = ok(combo)
        t = 0
        while not good and t < 150:
            combo = tuple(rng.choice(elig, size=n_components, replace=False))
            good, _ = ok(combo); t += 1
        if not good:
            continue
        best = score(combo)
        for _ in range(iters):
            cand = list(combo); cand[rng.integers(n_components)] = int(rng.choice(elig))
            if len(set(cand)) < n_components:
                continue
            cand = tuple(cand)
            g, _ = ok(cand)
            if not g:
                continue
            s = score(cand)
            if s > best:
                combo, best = cand, s
        if frozenset(combo) not in {frozenset(c) for c, _ in made}:
            made.append((combo, best))

    made.sort(key=lambda t: t[1], reverse=True)
    carrier_g = batch_g * (1 - aroma_load_pct / 100.0)
    aroma_g = batch_g * (aroma_load_pct / 100.0)

    out = []
    for rank, (combo, s) in enumerate(made, 1):
        nv = _novelty(combo, rows, X, ref_fp, ref_prof)
        prof_top = fp.top_descriptors(nv.pop("_profile"), desc_names)
        pct = _proportions([model.predict(X[i:i+1])[0] for i in combo])
        comps = []
        for i, p in zip(combo, pct):
            dv = rows[i]["descriptors"]
            notes = [desc_names[j] for j in np.argsort(dv)[::-1][:3] if dv[j] > 0] if dv is not None else []
            comps.append({"molecule": rows[i]["name"], "smiles": rows[i]["smiles"],
                          "cas": rows[i]["cas"], "percent_of_aroma_w_w": round(float(p), 2),
                          "mass_mg_in_batch": round(float(p) / 100.0 * aroma_g * 1000.0, 1),
                          "odor_notes": notes, "panel_pleasantness": round(rows[i]["pleasantness"], 1),
                          "safety": rows[i]["safety"]})
        comps.sort(key=lambda c: c["percent_of_aroma_w_w"], reverse=True)
        flavor_name = "-".join([d for d, _ in prof_top[:3]]) + " accord" if prof_top else "novel accord"
        out.append({"rank": rank, "flavor_name": flavor_name,
                    "predicted_mean_pleasantness": round(s, 2), "n_components": len(combo),
                    "flavor_profile_top_notes": prof_top, "novelty": nv,
                    "batch": {"total_g": batch_g, "aroma_g": round(aroma_g, 3),
                              "carrier_g": round(carrier_g, 3),
                              "carrier": "propylene glycol (food grade)",
                              "aroma_load_pct": aroma_load_pct},
                    "components": comps})

    return {
        "note": ("Flavor RECIPES (blends with proportions), scored by a real pleasantness "
                 "model (Keller 2016). Odor profiles from Leffingwell descriptors. Every "
                 "component passes a food-appropriate safety SCREEN (not a GRAS clearance). "
                 "Proportions are heuristic starting points to be balanced by a flavorist "
                 "with odor-threshold data. Do NOT ingest without regulatory + safety review."),
        "novelty_method": ("Each recipe is required to be distinct from every reference flavor "
                           "(chocolate, vanilla, coffee, banana, strawberry, rose) both "
                           "CHEMICALLY (mean best-match Tanimoto) and PERCEPTUALLY (descriptor-"
                           "profile cosine), and to share few components with any reference. "
                           "This demonstrates a flavor character not matching known/public "
                           "flavors; it does NOT prove absence from proprietary formulas."),
        "recipes": out,
    }


def to_markdown(suite):
    L = ["# Candidate novel flavor RECIPES (super-chocolate program)", "",
         f"> {suite['note']}", "", f"_Novelty:_ {suite['novelty_method']}", ""]
    for r in suite["recipes"]:
        prof = ", ".join(f"{d} ({v})" for d, v in r["flavor_profile_top_notes"]) or "n/a"
        L += [f"## Recipe {r['rank']}: \"{r['flavor_name']}\" - predicted pleasantness "
              f"{r['predicted_mean_pleasantness']}/100",
              "",
              f"Flavor profile (top notes): {prof}", "",
              f"Novelty: chemically {r['novelty']['chem_distance_to_nearest']} from nearest "
              f"({r['novelty']['nearest_chem_flavor']}); perceptually "
              f"{r['novelty']['perceptual_distance_to_nearest']} from nearest "
              f"({r['novelty']['nearest_perceptual_flavor']}); "
              f"{r['novelty']['components_resembling_known']}/{r['n_components']} components "
              f"resemble a known reference component.", "",
              f"Batch: {r['batch']['total_g']} g = {r['batch']['aroma_g']} g aroma in "
              f"{r['batch']['carrier_g']} g {r['batch']['carrier']}.", "",
              "| molecule | % aroma (w/w) | mg/batch | odor notes | CAS | pleasantness |",
              "|---|---|---|---|---|---|"]
        for c in r["components"]:
            L.append(f"| {c['molecule']} | {c['percent_of_aroma_w_w']} | {c['mass_mg_in_batch']} "
                     f"| {', '.join(c['odor_notes']) or 'n/a'} | {c['cas'] or 'n/a'} | "
                     f"{c['panel_pleasantness']} |")
        L.append("")
    L += ["## How to make a batch",
          "1. Weigh each aroma component (mg) into a clean amber vial in a fume hood.",
          "2. Add food-grade propylene glycol to the carrier mass shown.",
          "3. Cap, mix to dissolve, equilibrate 24-48 h.",
          "4. Evaluate by SMELL first, far below tasting levels. Do NOT taste without GRAS/FEMA "
          "confirmation of every component and use level (see HANDOFF.md).", "",
          "_Proportions are model-heuristic, not odor-activity-balanced; adjust by nose._"]
    return "\n".join(L)


if __name__ == "__main__":
    s = compose_recipes()
    print(to_markdown(s))
