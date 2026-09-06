"""Food-appropriate safety / toxicity screen for candidate flavor molecules.

WHY NOT BRENK/PAINS: those are drug-discovery assay-interference filters. Tested here,
they flag vanillin (a ubiquitous safe flavor) for its aldehyde yet pass benzene (a
carcinogen). They are miscalibrated for ingestion safety, so they are NOT used as the
verdict (they can be reported as a secondary note).

WHAT THIS DOES: a transparent hazard screen combining (1) a curated denylist of known
toxic / genotoxic / restricted substances, (2) structural alerts for genotoxic
toxicophores relevant to ingestion, and (3) basic property/element rules. It returns a
tiered verdict with reasons.

IMPORTANT: this is a SCREEN to keep obviously hazardous structures out of a candidate
suite, NOT a regulatory safety determination. Real clearance requires GRAS/FEMA review and
tox assays (Ames, hERG, subchronic). Nothing here authorizes eating anything.
"""

from __future__ import annotations

from rdkit import Chem

# (1) Known toxic / genotoxic / restricted, by canonical SMILES.
_DENY = {
    "benzene": "c1ccccc1",
    "formaldehyde": "C=O",
    "methanol": "CO",
    "acrylonitrile": "C=CC#N",
    "acrylamide": "C=CC(N)=O",
    "aniline": "Nc1ccccc1",
    "safrole": "C=CCc1ccc2OCOc2c1",
    "methyleugenol": "C=CCc1ccc(OC)c(OC)c1",
    "coumarin": "O=c1ccc2ccccc2o1",
    "hydrogen_cyanide": "C#N",
    "thujone": "CC1CCC2(C)C(=O)CC1C2",
}

# (2) Genotoxic structural alerts (high concern for ingestion). SMARTS.
_TOXICOPHORES = {
    "aromatic_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "N_nitroso": "[NX3][NX2]=O",
    "azo": "[#6]N=N[#6]",
    "aromatic_primary_amine": "[NX3;H2][c]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "aziridine": "[NX3r3]1[#6r3][#6r3]1",
    "hydrazine": "[NX3;!$(N=*)][NX3;!$(N=*)]",
    "diazo": "[#6]=[N+]=[N-]",
}

# (3) lower-concern alerts -> CAUTION, not TOXIC (many are legitimate flavor motifs).
_CAUTION = {
    "michael_acceptor_enal": "[CX3]=[CX3][CX3]=O",
    "isothiocyanate": "N=C=S",
    "isocyanate": "N=C=O",
    "alkyl_halide": "[CX4][Cl,Br,I]",
}

_ALLOWED_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl"}
_MW_MAX = 400.0


def _canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


_DENY_CANON = {v: k for k, v in ((k, _canon(s)) for k, s in _DENY.items()) if v}


def screen(smiles: str) -> dict:
    """Return {verdict: OK|CAUTION|TOXIC|INVALID, reasons: [...]}."""
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"verdict": "INVALID", "reasons": ["unparseable SMILES"]}

    reasons = []
    verdict = "OK"
    canon = Chem.MolToSmiles(mol)

    if canon in _DENY_CANON:
        return {"verdict": "TOXIC", "reasons": [f"denylist: {_DENY_CANON[canon]}"]}

    for name, smarts in _TOXICOPHORES.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            reasons.append(f"genotoxic alert: {name}")
            verdict = "TOXIC"
    if verdict == "TOXIC":
        return {"verdict": verdict, "reasons": reasons}

    # element / property rules
    elems = {a.GetSymbol() for a in mol.GetAtoms()}
    bad = elems - _ALLOWED_ELEMENTS
    if bad:
        reasons.append(f"non-food elements: {sorted(bad)}")
        verdict = "CAUTION"
    mw = Descriptors.MolWt(mol)
    if mw > _MW_MAX:
        reasons.append(f"MW {mw:.0f} > {_MW_MAX:.0f}")
        verdict = "CAUTION"

    for name, smarts in _CAUTION.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            reasons.append(f"caution motif: {name}")
            verdict = "CAUTION" if verdict != "TOXIC" else verdict

    if not reasons:
        reasons = ["no denylist/toxicophore/property flags (screen only, not GRAS review)"]
    return {"verdict": verdict, "reasons": reasons}


def is_safe(smiles: str) -> bool:
    """True only if the screen returns OK (CAUTION and TOXIC excluded from a suite)."""
    return screen(smiles)["verdict"] == "OK"


if __name__ == "__main__":
    tests = {
        "vanillin": "O=Cc1ccc(O)c(OC)c1",
        "ethyl butyrate": "CCCC(=O)OCC",
        "limonene": "CC(=C)C1CCC(C)=CC1",
        "benzene(toxic)": "c1ccccc1",
        "formaldehyde(toxic)": "C=O",
        "an epoxide(toxic)": "C1CO1",
        "safrole(toxic)": "C=CCc1ccc2OCOc2c1",
        "cinnamaldehyde(caution enal)": "O=C/C=C/c1ccccc1",
    }
    for n, s in tests.items():
        r = screen(s)
        print(f"{n:28s} -> {r['verdict']:8s} {r['reasons'][0]}")
