"""Existing flavor profiling, integrated as data (Leffingwell odor descriptors via Pyrfume).

Leffingwell labels 3522 molecules with 113 binary odor descriptors (fruity, cocoa, floral,
green, ...). We use it to:
  - attach a real odor profile to each molecule in a recipe;
  - build a descriptor profile for each recipe (fraction of components carrying each note);
  - define a NOVEL flavor concretely: a recipe whose descriptor profile is distinct (cosine)
    from every reference natural/known flavor's descriptor profile. This is a data-grounded
    operationalization of "a flavor character not matching known flavors". It cannot prove
    absolute absence in nature, but it shows the blend's perceived character is not one of
    the known references.
"""

from __future__ import annotations

import numpy as np


def load_descriptor_table():
    """Return (descriptor_names, cid_to_vector dict) from Leffingwell."""
    import pyrfume
    beh = pyrfume.load_data("leffingwell/behavior.csv")
    names = list(beh.columns)
    M = beh.to_numpy(dtype=float)
    cids = list(beh.index)
    return names, {int(c): M[i] for i, c in enumerate(cids)}


def canonical_smiles_index():
    """Map canonical SMILES -> descriptor vector, for reference molecules given by SMILES."""
    import pyrfume
    from rdkit import Chem
    beh = pyrfume.load_data("leffingwell/behavior.csv")
    mol = pyrfume.load_data("leffingwell/molecules.csv")
    names = list(beh.columns)
    out = {}
    for cid, row in mol.iterrows():
        try:
            m = Chem.MolFromSmiles(row["IsomericSMILES"])
            if m is None or int(cid) not in beh.index:
                continue
            out[Chem.MolToSmiles(m)] = beh.loc[cid].to_numpy(dtype=float)
        except Exception:
            continue
    return names, out


def recipe_profile(vectors):
    """Mean descriptor vector over components (fraction carrying each note)."""
    V = [v for v in vectors if v is not None]
    if not V:
        return None
    return np.mean(np.vstack(V), axis=0)


def cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(a @ b / (na * nb))


def top_descriptors(profile, names, k=8, min_frac=0.15):
    if profile is None:
        return []
    idx = np.argsort(profile)[::-1]
    return [(names[i], round(float(profile[i]), 2)) for i in idx[:k] if profile[i] >= min_frac]
