"""Receptor layer: molecule -> odorant-receptor activation (the missing middle in the chain).

Wintermute's chain is molecule -> receptor activation profile -> flavor. Everything else in
this repo learns structure->percept directly; this module grounds the intermediate on REAL
data: Mainland et al. 2015, a human odorant-receptor screen (via Pyrfume), which gives
confirmed OR-ligand activations with EC50.

Open receptor-activation data is sparse (a few dozen confirmed OR-ligand pairs across ~100
molecules), far from the ~400 human ORs, so this is a real but partial receptor layer, not
a full molecule->400-OR model (that needs a larger resource like M2OR). It provides:
  - known_receptors(cid): ORs a molecule is confirmed to activate (Mainland), with EC50.
  - predict_receptors(smiles): structural nearest-neighbour inference of likely OR
    activation, validated leave-one-out against chance.

Refs: Mainland et al. 2015 (Nat Neurosci); receptor-response-predicts-percept: Lee/Qian
2023 (POM). Retrieved via Pyrfume.
"""

from __future__ import annotations

import numpy as np

from featurize import smiles_to_ecfp4


def _tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def load_mainland():
    """Return list of {cid, smiles, or_gene, ec50, fp} confirmed activating pairs."""
    import pyrfume
    from rdkit import Chem
    beh = pyrfume.load_data("mainland_2015/behavior.csv")
    mol = pyrfume.load_data("mainland_2015/molecules.csv")
    # map by molecule name (behavior.OdorName <-> molecules.name), case-insensitive
    name2 = {str(r["name"]).strip().lower(): (int(c), r["IsomericSMILES"])
             for c, r in mol.iterrows()}
    pairs = []
    for _, r in beh.iterrows():
        key = str(r.get("OdorName", "")).strip().lower()
        if key not in name2:
            continue
        cid, smi = name2[key]
        if not smi or Chem.MolFromSmiles(smi) is None:
            continue
        try:
            fp = smiles_to_ecfp4(smi)
        except Exception:
            continue
        pairs.append({"cid": cid, "smiles": smi, "or_gene": str(r.get("Gene", "")),
                      "ec50": str(r.get("EC50", "")), "fp": fp})
    return pairs


def known_receptors(cid, pairs):
    return [{"or_gene": p["or_gene"], "ec50": p["ec50"]} for p in pairs if p["cid"] == cid]


def predict_receptors(smiles, pairs, k=1, min_sim=0.2):
    """Infer likely OR activation from the k structurally nearest confirmed ligands."""
    try:
        fp = smiles_to_ecfp4(smiles)
    except Exception:
        return []
    scored = sorted(((p, _tanimoto(fp, p["fp"])) for p in pairs),
                    key=lambda t: t[1], reverse=True)
    out = []
    for p, sim in scored[:k]:
        if sim >= min_sim:
            out.append({"or_gene": p["or_gene"], "via_similarity_to": p["smiles"],
                        "tanimoto": round(sim, 3)})
    return out


def validate_leave_one_out(pairs):
    """For each ligand, does its nearest OTHER ligand share an activated OR more often than a
    random pair would? Reports NN-shared-OR rate vs the chance baseline."""
    genes = [p["or_gene"] for p in pairs]
    n = len(pairs)
    nn_hits = 0
    for i in range(n):
        sims = [(_tanimoto(pairs[i]["fp"], pairs[j]["fp"]), j) for j in range(n) if j != i]
        _, j = max(sims)
        nn_hits += int(pairs[i]["or_gene"] == pairs[j]["or_gene"])
    nn_rate = nn_hits / n
    # chance: probability two random pairs share the same OR gene
    from collections import Counter
    c = Counter(genes)
    chance = sum((v / n) ** 2 for v in c.values())
    return {"n_pairs": n, "nn_shared_or_rate": round(nn_rate, 3),
            "chance_rate": round(chance, 3), "lift": round(nn_rate / chance, 2) if chance else None}


if __name__ == "__main__":
    pairs = load_mainland()
    print("confirmed OR-ligand pairs:", len(pairs))
    print("validation:", validate_leave_one_out(pairs))
    print("example predict (eugenol):", predict_receptors("C=CCc1ccc(O)c(OC)c1", pairs))
