"""Offline unit tests for the real-molecule flavor pipeline (no network needed)."""

import numpy as np

from featurize import smiles_to_ecfp4, tanimoto
from safety import screen, is_safe


def test_featurizer():
    fp = smiles_to_ecfp4("O=Cc1ccc(O)c(OC)c1")  # vanillin
    assert fp.shape == (2048,) and fp.sum() > 0
    assert tanimoto(fp, fp) == 1.0


def test_safety_flags_toxic():
    for smi in ["c1ccccc1", "C=O", "C1CO1", "C=CCc1ccc2OCOc2c1"]:  # benzene, formaldehyde, epoxide, safrole
        assert screen(smi)["verdict"] == "TOXIC"


def test_safety_passes_flavors():
    for smi in ["O=Cc1ccc(O)c(OC)c1", "CCCC(=O)OCC", "CC(=C)C1CCC(C)=CC1"]:  # vanillin, ethyl butyrate, limonene
        assert is_safe(smi)


def test_safety_caution_not_ok():
    # cinnamaldehyde (enal) should be CAUTION, hence excluded from a suite
    assert screen("O=C/C=C/c1ccccc1")["verdict"] == "CAUTION"
    assert not is_safe("O=C/C=C/c1ccccc1")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("PASS", n)
    print("all tests passed")
