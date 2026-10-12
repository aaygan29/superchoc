"""Can the receptor layer enter J? Measure what links it to the percept in this repo's data.
Writes results/newslate/receptor_in_J_check.json."""
import json, os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from real_data import build_keller_pleasantness
from rdkit import Chem
from rdkit.Chem import inchi
pairs = pd.read_csv(os.path.join(HERE, "..", "..", "..", "data", "raw", "m2or_pairs.csv"))
df = build_keller_pleasantness()
ik = lambda s: inchi.MolToInchiKey(Chem.MolFromSmiles(s))
pairs["ik"] = pairs.smiles.map(ik); df["ik"] = df.smiles.map(ik)
mol = pairs.groupby("ik").agg(n_tested=("resp", "size"), n_resp=("resp", "sum")).reset_index()
m = mol.merge(df[["ik", "pleasantness"]], on="ik")
out = {"m2or_molecules": int(mol.shape[0]), "m2or_receptors": int(pairs.uniprot.nunique()), "n_tested_pairs": int(len(pairs)),
       "m2or_molecules_with_keller_pleasantness_label": int(len(m)),
       "m2or_molecules_with_zero_responsive_receptors": int((mol.n_resp == 0).sum())}
if len(m) >= 8:
    r = spearmanr(m.n_resp, m.pleasantness)
    out["spearman_n_responsive_receptors_vs_pleasantness"] = {"rho": float(r.correlation), "p": float(r.pvalue), "n": int(len(m))}
out["verdict"] = ("The receptor layer cannot enter J as a validated term: (1) the only receptor-to-percept evidence available is the overlap above; "
                  "(2) the 1.57x NN-shared-receptor statistic is a ligand-similarity check on 22 pairs (2 hits = 1 mutual pair), not a score; "
                  "(3) the M2OR model scores (molecule, receptor) activation, and J would need an unvalidated map from activation profile to pleasantness. "
                  "Any weight on it would be arbitrary, so the receptor claim is dropped from the contribution list.")
json.dump(out, open(os.path.join(HERE, "..", "results", "newslate", "receptor_in_J_check.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
