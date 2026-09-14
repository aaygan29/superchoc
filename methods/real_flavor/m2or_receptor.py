"""Molecule -> odorant-receptor activation model, trained on real M2OR data.

Upgrades the sparse nearest-neighbour receptor layer (receptors.py, ~1.6x chance) to a learned
predictor: given a molecule and a human odorant receptor (its amino-acid sequence), predict
whether the molecule ACTIVATES that receptor. This is the biologically correct middle of
the molecule -> receptor activation -> percept chain.

Data: M2OR (Lalis et al. 2024, Nucleic Acids Research; github.com/chemosim-lab/M2OR), the curated
odorant-receptor--molecule database. We use the human subset with a binarized Responsive label and
a receptor sequence, resolving molecule CIDs to SMILES via Pyrfume. The committed, reproducible
subset is data/raw/m2or_pairs.csv (uniprot, cid, smiles, resp) + data/raw/m2or_receptors.csv
(uniprot -> sequence): 92 molecules x 375 human ORs, 2340 tested pairs, ~5% positive.

Features: molecule ECFP4 (2048 bits) concatenated with the receptor's amino-acid composition
(20-dim), so the model can score any (molecule, receptor-sequence) pair, including receptors not
seen with that molecule. Model: random forest (class-balanced).

Validation: LEAVE-MOLECULE-OUT grouped CV (hold out all pairs of a molecule, predict its receptor
activations from structure + receptor sequence), pooled ROC-AUC vs the ~5% base rate. This is the
honest test of "new molecule -> which receptors does it hit". Coverage is modest (92 molecules),
so this is a real but partial model, reported with its scope.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from featurize import smiles_to_ecfp4

_HERE = os.path.dirname(__file__)
_DATA = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "raw"))
_AA = "ACDEFGHIKLMNPQRSTVWY"


def _aac(seq):
    """Amino-acid composition (20-dim) of a receptor sequence."""
    seq = "".join(c for c in str(seq).upper() if c in _AA)
    if not seq:
        return np.zeros(len(_AA))
    v = np.array([seq.count(a) for a in _AA], dtype=float)
    return v / v.sum()


def load():
    pairs = pd.read_csv(os.path.join(_DATA, "m2or_pairs.csv"))
    recs = pd.read_csv(os.path.join(_DATA, "m2or_receptors.csv"))
    recs = recs.dropna(subset=["uniprot", "sequence"])          # drop any missing-accession rows
    pairs = pairs[pairs["uniprot"].isin(set(recs["uniprot"]))]  # keep pairs with a known receptor
    seq = dict(zip(recs["uniprot"], recs["sequence"]))
    return pairs, seq


def _features(pairs, seq):
    X, y, groups = [], [], []
    aac_cache = {u: _aac(s) for u, s in seq.items()}
    fp_cache = {}
    for _, r in pairs.iterrows():
        s = r["smiles"]
        if s not in fp_cache:
            try:
                fp_cache[s] = smiles_to_ecfp4(s)
            except Exception:
                fp_cache[s] = None
        fp = fp_cache[s]
        a = aac_cache.get(r["uniprot"])
        if fp is None or a is None:
            continue
        X.append(np.concatenate([fp, a])); y.append(int(r["resp"])); groups.append(int(r["cid"]))
    return np.vstack(X), np.array(y), np.array(groups)


class M2ORReceptorModel:
    def __init__(self):
        self.pairs, self.seq = load()
        self.X, self.y, self.groups = _features(self.pairs, self.seq)
        self.aac = {u: _aac(s) for u, s in self.seq.items()}
        self.model = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                            random_state=0, n_jobs=-1).fit(self.X, self.y)

    def cross_validate(self, seed=0):
        """Leave-molecule-out grouped CV, pooled ROC-AUC vs base rate."""
        n_groups = len(np.unique(self.groups))
        gkf = GroupKFold(n_splits=min(10, n_groups))
        preds = np.zeros(len(self.y))
        for tr, te in gkf.split(self.X, self.y, self.groups):
            m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                       random_state=seed, n_jobs=-1).fit(self.X[tr], self.y[tr])
            preds[te] = m.predict_proba(self.X[te])[:, 1]
        return {"n_pairs": int(len(self.y)), "n_molecules": int(n_groups),
                "n_receptors": int(self.pairs["uniprot"].nunique()),
                "positives": int(self.y.sum()), "base_rate": round(float(self.y.mean()), 3),
                "leave_molecule_out_auc": round(float(roc_auc_score(self.y, preds)), 3)}

    def activation_profile(self, smiles, top=10):
        """Predicted P(activate) across all M2OR human receptors for a new molecule."""
        try:
            fp = smiles_to_ecfp4(smiles)
        except Exception:
            return None
        rows = [(u, np.concatenate([fp, a])) for u, a in self.aac.items()]
        P = self.model.predict_proba(np.vstack([x for _, x in rows]))[:, 1]
        order = np.argsort(P)[::-1][:top]
        us = [rows[i][0] for i in order]
        return [{"uniprot": u, "p_activate": round(float(P[i]), 3)} for i, u in zip(order, us)]


if __name__ == "__main__":
    m = M2ORReceptorModel()
    cv = m.cross_validate()
    print("M2OR receptor model (leave-molecule-out):", json.dumps(cv, indent=2))
    # demo: which receptors does vanillin most likely hit?
    prof = m.activation_profile("O=Cc1ccc(O)c(OC)c1", top=5)
    print("vanillin top predicted receptors:", prof)
    json.dump(cv, open(os.path.join(_HERE, "results", "m2or_receptor.json"), "w"), indent=2)
    print("wrote results/m2or_receptor.json")
