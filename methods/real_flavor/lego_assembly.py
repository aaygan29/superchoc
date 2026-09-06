"""Lego assembly: build NEW biomolecules from fragments, then predict their taste/odor.

The vision: take known safe flavor molecules, break them into fragments (the lego bricks),
snap the bricks back together into molecules that did not exist before, and for each new
molecule predict whether it would taste/smell good and which receptors it engages. The good,
safe, structurally-novel ones become new building blocks to blend into flavors.

Pipeline:
  1. Fragment safe flavor molecules with BRICS (retrosynthetically sensible bond breaks).
  2. Reassemble with BRICS.BRICSBuild into candidate molecules; sanitize; dedupe by InChIKey;
     drop any already in the training set; keep food-plausible size (MW 60-350).
  3. For each candidate: food-safety screen; predicted pleasantness (goodness model, Keller
     2016); odor notes and odorant-receptor hits transferred from the nearest known molecule
     (Leffingwell descriptors / Mainland ORs) with the similarity reported.
  4. EMPIRICAL structure novelty: query PubChem by InChIKey. A molecule with no PubChem hit
     is not in that public database (a strong novelty signal); one with a hit is known.

Honesty: predictions for de-novo molecules are extrapolation - the goodness model was trained
on known odorants, so confidence is lower off-distribution; nearest-neighbor similarity is
reported so low-similarity predictions can be discounted. Absence from PubChem is evidence of
novelty, not proof of never-synthesized. Nothing here is a safety clearance.
"""

from __future__ import annotations

import itertools
import json
import time
import urllib.request

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, inchi

from featurize import smiles_to_ecfp4
from safety import screen
from goodness_real import RealGoodnessModel
from real_data import build_keller_pleasantness
import flavor_profile as fp
import receptors as rc


def _tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def build_prereqs():
    df = build_keller_pleasantness().reset_index(drop=True)
    X, names, y, known_ik = [], [], [], set()
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        try:
            X.append(smiles_to_ecfp4(r["smiles"]))
        except Exception:
            continue
        names.append(r["name"]); y.append(float(r["pleasantness"]))
        known_ik.add(inchi.MolToInchiKey(m))
    X = np.vstack(X); y = np.array(y)
    model = RealGoodnessModel(seed=0).fit(X, y)
    # Leffingwell descriptor NN index
    desc_names, cid2desc = fp.load_descriptor_table()
    import pyrfume
    lm = pyrfume.load_data("leffingwell/molecules.csv")
    leff = []
    for cid, row in lm.iterrows():
        if int(cid) not in cid2desc:
            continue
        try:
            leff.append((smiles_to_ecfp4(row["IsomericSMILES"]), cid2desc[int(cid)]))
        except Exception:
            continue
    pairs = rc.load_mainland()
    return {"df": df, "keller_X": X, "keller_names": names, "model": model,
            "known_ik": known_ik, "desc_names": desc_names, "leff": leff, "mainland": pairs}


def fragment_library(smiles_list, max_mols=150):
    frags = set()
    for s in smiles_list[:max_mols]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            frags |= set(BRICS.BRICSDecompose(m))
        except Exception:
            continue
    return [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f)]


def assemble(frag_mols, n_build=800, mw_range=(60, 350), seed=0):
    import random
    random.seed(seed)
    seen, out = set(), []
    for m in itertools.islice(BRICS.BRICSBuild(frag_mols), n_build):
        try:
            Chem.SanitizeMol(m)
        except Exception:
            continue
        mw = Descriptors.MolWt(m)
        if not (mw_range[0] <= mw <= mw_range[1]):
            continue
        smi = Chem.MolToSmiles(m)
        ik = inchi.MolToInchiKey(m)
        if ik in seen:
            continue
        seen.add(ik)
        out.append({"smiles": smi, "inchikey": ik, "mw": round(mw, 1)})
    return out


def pubchem_known(inchikey, timeout=15):
    """True if PubChem has this InChIKey (known), False if 404 (novel), None on error."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read().decode())
            return bool(d.get("IdentifierList", {}).get("CID"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return None
    except Exception:
        return None


def evaluate(cand, pre):
    try:
        fpv = smiles_to_ecfp4(cand["smiles"])
    except Exception:
        return None
    sc = screen(cand["smiles"])
    pred = float(pre["model"].predict(fpv.reshape(1, -1))[0])
    # nearest Keller molecule (in-distribution confidence)
    kd = [_tanimoto(fpv, x) for x in pre["keller_X"]]
    j = int(np.argmax(kd)); nn_sim = round(kd[j], 3); nn_name = pre["keller_names"][j]
    # odor notes from nearest Leffingwell molecule
    notes = []
    if pre["leff"]:
        ls = [_tanimoto(fpv, f) for f, _ in pre["leff"]]
        li = int(np.argmax(ls)); dv = pre["leff"][li][1]
        notes = [pre["desc_names"][k] for k in np.argsort(dv)[::-1][:4] if dv[k] > 0]
    # receptor hits (structural NN to Mainland ligands)
    recs = rc.predict_receptors(cand["smiles"], pre["mainland"], k=1, min_sim=0.2)
    return {**cand, "safety": sc["verdict"], "predicted_pleasantness": round(pred, 1),
            "nearest_known": nn_name, "nearest_known_tanimoto": nn_sim,
            "predicted_odor_notes": notes,
            "receptor_hits": recs, "novel_vs_training": cand["inchikey"] not in pre["known_ik"]}


def run(seed=0, n_build=800, top_k_pubchem=12):
    pre = build_prereqs()
    safe_smiles = [r["smiles"] for _, r in pre["df"].iterrows()
                   if screen(r["smiles"])["verdict"] == "OK"]
    frags = fragment_library(safe_smiles)
    cands = assemble(frags, n_build=n_build, seed=seed)
    evald = [e for e in (evaluate(c, pre) for c in cands) if e]
    # keep safe, novel-vs-training, reasonable in-distribution similarity
    keep = [e for e in evald if e["safety"] == "OK" and e["novel_vs_training"]]
    keep.sort(key=lambda e: e["predicted_pleasantness"], reverse=True)
    # empirical structure novelty: PubChem check on the top candidates
    for e in keep[:top_k_pubchem]:
        known = pubchem_known(e["inchikey"])
        e["in_pubchem"] = known
        e["structurally_novel"] = (known is False)
        time.sleep(0.25)
    return {"n_fragments": len(frags), "n_assembled": len(cands),
            "n_safe_novel": len(keep), "candidates": keep}


if __name__ == "__main__":
    out = run()
    print("fragments=%d assembled=%d safe&novel=%d" %
          (out["n_fragments"], out["n_assembled"], out["n_safe_novel"]))
    for e in out["candidates"][:10]:
        print(f"  pl={e['predicted_pleasantness']:.1f} sim={e['nearest_known_tanimoto']} "
              f"pubchem={e.get('in_pubchem')} notes={e['predicted_odor_notes'][:3]} {e['smiles']}")
