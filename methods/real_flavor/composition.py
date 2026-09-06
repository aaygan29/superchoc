"""Compositional statistics: which chemical classes make a food taste good, and in what
proportions, learned from real data and used to shape new recipes.

We classify each molecule into functional/chemical classes (ester, aldehyde, lactone,
pyrazine, phenol, thiol, acid, amine, ...) with RDKit SMARTS, then measure from the real
Keller 2016 pleasantness data which classes are associated with high vs low pleasantness
(the "critical components" and their effect). A recipe's composition is the fraction of its
components carrying each class. We derive a TARGET composition (emphasize high-pleasantness
classes, avoid unpleasant ones) and score/steer recipes toward it.

Mathematical backbone (Lean-checked, FlavorMath.lean): under a linear per-class score, the
blended score of any composition is bounded by the best single class score (weighted average
<= max). So composition tuning alone cannot beat the best class; exceeding it needs
non-additive interactions (the learned mixture model) - the same wall, now at the class level.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

CLASS_SMARTS = {
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "lactone": "[CX3](=O)[OX2;R][#6;R]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H][c]",
    "ether": "[OD2]([#6])[#6]",
    "pyrazine": "c1cnccn1",
    "furan": "c1ccoc1",
    "sulfur": "[#16]",
    "amine": "[NX3;!$(N=O);!$(N=*)]",
    "alkene": "[CX3]=[CX3]",
    "aromatic": "c1ccccc1",
    "terpenoid_like": "[CX4]([CH3])([CH3])",
}

_PATTS = {k: Chem.MolFromSmarts(v) for k, v in CLASS_SMARTS.items()}


def classify(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return set()
    return {k for k, p in _PATTS.items() if p is not None and m.HasSubstructMatch(p)}


def class_pleasantness_stats(df):
    """For each class: mean pleasantness of molecules that HAVE it vs the overall mean, n."""
    classes = {k: [] for k in CLASS_SMARTS}
    overall = []
    for _, r in df.iterrows():
        cs = classify(r["smiles"]); pl = float(r["pleasantness"]); overall.append(pl)
        for k in cs:
            classes[k].append(pl)
    om = float(np.mean(overall))
    out = {}
    for k, vals in classes.items():
        if len(vals) >= 5:
            out[k] = {"n": len(vals), "mean_pleasantness": round(float(np.mean(vals)), 1),
                      "lift_vs_overall": round(float(np.mean(vals)) - om, 1)}
    return {"overall_mean": round(om, 1), "by_class": out}


def composition_vector(smiles_list):
    """Fraction of components carrying each class (multi-label)."""
    n = len(smiles_list)
    counts = {k: 0 for k in CLASS_SMARTS}
    for s in smiles_list:
        for k in classify(s):
            counts[k] += 1
    return {k: counts[k] / n for k in CLASS_SMARTS} if n else counts


def target_composition(stats, top=5):
    """Target = the classes with the highest positive pleasantness lift."""
    by = stats["by_class"]
    ranked = sorted(by.items(), key=lambda kv: kv[1]["lift_vs_overall"], reverse=True)
    good = [k for k, v in ranked if v["lift_vs_overall"] > 0][:top]
    bad = [k for k, v in ranked if v["lift_vs_overall"] < 0][-top:]
    return {"favor": good, "avoid": bad}


def composition_score(smiles_list, target):
    """Reward components in favored classes, penalize avoided classes (mean over components)."""
    if not smiles_list:
        return 0.0
    s = 0.0
    for smi in smiles_list:
        cs = classify(smi)
        s += sum(1 for k in cs if k in target["favor"]) - sum(1 for k in cs if k in target["avoid"])
    return round(s / len(smiles_list), 3)


if __name__ == "__main__":
    import json
    from real_data import build_keller_pleasantness
    df = build_keller_pleasantness()
    stats = class_pleasantness_stats(df)
    tgt = target_composition(stats)
    print("OVERALL mean pleasantness:", stats["overall_mean"])
    ranked = sorted(stats["by_class"].items(), key=lambda kv: kv[1]["lift_vs_overall"], reverse=True)
    for k, v in ranked:
        print(f"  {k:16s} n={v['n']:3d} mean={v['mean_pleasantness']:5.1f} lift={v['lift_vs_overall']:+.1f}")
    print("TARGET favor:", tgt["favor"], "| avoid:", tgt["avoid"])
