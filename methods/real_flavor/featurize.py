"""RDKit molecular featurization: SMILES -> ECFP4 (Morgan r=2) bit vectors."""

from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=20000)
def smiles_to_ecfp4(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def featurize(smiles_list, n_bits: int = 2048, radius: int = 2):
    """Return (X, valid_mask): features for parseable SMILES, mask of which parsed."""
    rows, mask = [], []
    for s in smiles_list:
        try:
            rows.append(smiles_to_ecfp4(s, n_bits, radius))
            mask.append(True)
        except Exception:
            mask.append(False)
    return np.vstack(rows), np.array(mask)


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
