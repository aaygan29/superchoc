"""Real molecular featurization (the real-data entry point).

Turns a SMILES string into an ECFP4 fingerprint (Morgan radius 2), the standard
structural feature behind the chemoinformatic odor models in docs/LITERATURE.md
(e.g. the DREAM baseline). This is the bridge from real molecules to the goodness
model: swap the synthetic FlavorSpace embedding for these fingerprints once a real
labeled dataset (see data/) is wired in.

RDKit is required only for this module. The synthetic validation pipeline does not
import it, so the method can be validated without any chemistry stack.
"""

from __future__ import annotations

import numpy as np


def smiles_to_ecfp4(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Return an ECFP4 bit vector for a SMILES, or raise ValueError if unparseable."""
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


def featurize_smiles_list(smiles_list, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Stack ECFP4 features for a list of SMILES into an (n, n_bits) matrix."""
    return np.vstack([smiles_to_ecfp4(s, n_bits, radius) for s in smiles_list])


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprints (novelty metric)."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
