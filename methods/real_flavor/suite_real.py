"""Compose a suite of novel, safety-screened flavor combinations from REAL molecules.

Pipeline (all real inputs):
  1. Real molecules + real pleasantness labels (Keller 2016) via real_data.py.
  2. Real goodness model (ECFP4 -> pleasantness) trained on all labeled molecules.
  3. Food-appropriate safety screen (safety.py); only OK molecules are eligible.
  4. Compose k-molecule combos, hill-climb on mean predicted pleasantness, keep only
     combos chemically NOVEL vs a cocoa key-odorant reference set (mean Tanimoto low).
  5. Emit a ranked suite (JSON + markdown) with member names/SMILES, predicted pleasantness,
     per-member safety, novelty, and a lab/chef handoff note.

Combo score = mean predicted pleasantness of members (a transparent additive estimate).
The learned mixture model (mixture_model.py) is the documented refinement for non-additive
effects; it is validated separately and not yet trained on real mixture labels, so it is
not used to rank here.

Cocoa reference odorants (literature key cocoa aroma compounds), used only for novelty:
"""

from __future__ import annotations

import json

import numpy as np

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from goodness_real import RealGoodnessModel
from safety import screen

# Representative cocoa/chocolate key aroma compounds (for novelty reference only).
COCOA_REFERENCE = {
    "3-methylbutanal": "CC(C)CC=O",
    "2-methylbutanal": "CCC(C)C=O",
    "phenylacetaldehyde": "O=CCc1ccccc1",
    "2-phenylethanol": "OCCc1ccccc1",
    "linalool": "CC(C)=CCCC(C)(O)C=C",
    "2,3,5-trimethylpyrazine": "Cc1cnc(C)c(C)n1",
}


def _tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def build():
    df = build_keller_pleasantness()
    # featurize + safety screen
    feats, keep = [], []
    for _, row in df.iterrows():
        try:
            fp = smiles_to_ecfp4(row["smiles"])
        except Exception:
            continue
        verdict = screen(row["smiles"])["verdict"]
        feats.append(fp)
        keep.append({"name": row["name"], "smiles": row["smiles"],
                     "pleasantness": float(row["pleasantness"]), "safety": verdict})
    X = np.vstack(feats)
    y = np.array([k["pleasantness"] for k in keep])
    model = RealGoodnessModel(seed=0).fit(X, y)

    # eligible = safety OK only
    eligible = [i for i, k in enumerate(keep) if k["safety"] == "OK"]
    ref_fps = np.vstack([smiles_to_ecfp4(s) for s in COCOA_REFERENCE.values()])
    return df, X, keep, model, eligible, ref_fps


def novelty_from_cocoa(fp, ref_fps):
    return 1.0 - max(_tanimoto(fp, r) for r in ref_fps)  # 1 - max sim = min distance


def combo_pred(model, X, combo):
    return float(model.predict(X[list(combo)]).mean())


def compose_suite(seed=0, k=4, n_candidates=10, iters=400, novelty_min=0.5):
    df, X, keep, model, eligible, ref_fps = build()
    rng = np.random.default_rng(seed)
    elig = np.array(eligible)

    def novelty(combo):
        return float(np.mean([novelty_from_cocoa(X[i], ref_fps) for i in combo]))

    results = []
    for _ in range(n_candidates):
        combo = tuple(rng.choice(elig, size=k, replace=False))
        best = combo_pred(model, X, combo) if novelty(combo) >= novelty_min else -np.inf
        tries = 0
        while best == -np.inf and tries < 100:
            combo = tuple(rng.choice(elig, size=k, replace=False))
            if novelty(combo) >= novelty_min:
                best = combo_pred(model, X, combo)
            tries += 1
        for _ in range(iters):
            cand = list(combo); cand[rng.integers(k)] = int(rng.choice(elig))
            if len(set(cand)) < k:
                continue
            cand = tuple(cand)
            if novelty(cand) < novelty_min:
                continue
            s = combo_pred(model, X, cand)
            if s > best:
                combo, best = cand, s
        results.append({
            "members": [keep[i]["name"] for i in combo],
            "smiles": [keep[i]["smiles"] for i in combo],
            "predicted_mean_pleasantness": round(best, 2),
            "member_safety": [keep[i]["safety"] for i in combo],
            "novelty_from_cocoa": round(novelty(combo), 3),
        })
    results.sort(key=lambda r: r["predicted_mean_pleasantness"], reverse=True)
    for n, r in enumerate(results):
        r["rank"] = n + 1

    n_ok = sum(1 for kk in keep if kk["safety"] == "OK")
    suite = {
        "note": ("Predicted pleasantness from a REAL model (Keller 2016 human panel labels, "
                 "ECFP4 -> RandomForest). Members are real molecules and all pass a food-"
                 "appropriate safety SCREEN (not a GRAS determination). Scores are combo "
                 "mean predicted pleasantness (additive estimate); a learned mixture model "
                 "is the documented non-additive refinement. Do NOT ingest without proper "
                 "safety review."),
        "n_molecules_total": len(keep),
        "n_molecules_safe_eligible": n_ok,
        "pleasantness_scale": "0-100 (human panel)",
        "candidates": results,
    }
    return suite


def to_markdown(suite):
    L = ["# Candidate super-chocolate flavor suite (real molecules)", "",
         f"> {suite['note']}", "",
         f"Eligible safe molecules: {suite['n_molecules_safe_eligible']} / "
         f"{suite['n_molecules_total']}. Pleasantness scale: {suite['pleasantness_scale']}.",
         "", "| rank | molecules to combine | predicted pleasantness | novelty vs cocoa | safety |",
         "|---|---|---|---|---|"]
    for c in suite["candidates"]:
        L.append(f"| {c['rank']} | {', '.join(c['members'])} | "
                 f"{c['predicted_mean_pleasantness']} | {c['novelty_from_cocoa']} | "
                 f"{'/'.join(sorted(set(c['member_safety'])))} |")
    L += ["", "SMILES for each candidate are in the JSON, for a lab or molecular "
          "gastronomist to source and blend after independent safety review."]
    return "\n".join(L)


if __name__ == "__main__":
    s = compose_suite()
    print(to_markdown(s))
