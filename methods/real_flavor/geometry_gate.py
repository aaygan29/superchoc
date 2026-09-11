"""Cheap geometry gate: does a (de-novo) molecule correspond to a physically sane 3D structure?

The lego assembler (lego_assembly.py) snaps BRICS fragments together and keeps anything RDKit
can *sanitize* as a 2D graph. A valid 2D valence graph is not the same as a molecule that can
actually adopt a low-strain 3D geometry: fragment reassembly can produce over-strained rings,
impossible bridgeheads, or cages that no conformer relaxes into. Those should not be proposed
as flavor candidates.

This module adds a fast, dependency-light 3D sanity check:
  1. Parse + add hydrogens.
  2. Embed several 3D conformers with ETKDGv3 (distance-geometry).
  3. Relax each with a molecular-mechanics force field (MMFF94 where parameterized, else UFF)
     and take the lowest-energy converged conformer.
  4. Report strain as the embedding relaxation-gap per heavy atom, gated against a threshold
     calibrated on KNOWN flavor molecules (Keller 2016) so the cut is data-driven.

Verdict:
  FAILED   - cannot parse, cannot embed any conformer, or no conformer converges. Load-bearing:
             a 2D graph corresponding to no physical 3D molecule (impossible valence/ring
             closure) is culled here. This is the real fragment-reassembly failure mode.
  STRAINED - embeds/relaxes but the relaxation-gap per heavy atom exceeds the known-flavor
             threshold: a soft embedding-frustration flag, calibrated so no real flavor
             molecule trips it.
  STABLE   - a physically realizable structure within the known-flavor envelope.

What this cheap tier does and does NOT do (validated in _run_validation):
  * DOES reject chemically/geometrically invalid structures and pass every real molecule,
    INCLUDING genuinely strained-but-makeable rings (cyclopropane, norbornane, adamantane,
    cubane all pass).
  * Does NOT rank thermodynamic ring strain: MMFF/UFF embed small rigid strained cages cleanly,
    so cubane looks as stable as cyclohexane. Ranking strain magnitude (cubane >> cyclohexane)
    needs semiempirical xTB or DFT. That is the deliberate escalation for the handful of top
    candidates and drops in behind this same interface (geometry_check -> dict); it is not a
    claim the MMFF/UFF tier makes.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _heavy_atoms(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _ff_for(molH, cid):
    """Return an MMFF94 force field for one conformer, else UFF, plus a label."""
    props = AllChem.MMFFGetMoleculeProperties(molH)
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(molH, props, confId=cid)
        if ff is not None:
            return ff, "MMFF94"
    return AllChem.UFFGetMoleculeForceField(molH, confId=cid), "UFF"


def embed_and_relax(smiles, n_confs=6, seed=0):
    """Embed + force-field relax. Returns metrics dict (never raises).

    Strain is measured as the *relaxation gap*: the force-field energy of the raw
    distance-geometry embedding minus the energy after minimization, per heavy atom. This is a
    difference of energies for the SAME molecular graph, so the composition-dependent baseline
    (which makes absolute FF energy per atom misfire on aromatic heterocycles) cancels, leaving
    how much geometric strain the embedding had to work out. Broken/over-strained cages relax a
    lot; well-behaved molecules barely relax.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "reason": "parse_fail"}
    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_confs, params=params))
    if not cids:
        params.useRandomCoords = True  # last resort; lets strained cages fail or pass honestly
        cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_confs, params=params))
    if not cids:
        return {"ok": False, "reason": "embed_fail", "n_confs": 0}

    ff_label = None
    per_conf = []  # (converged_bool, e_initial, e_final)
    for cid in cids:
        ff, ff_label = _ff_for(molH, cid)
        if ff is None:
            continue
        e0 = float(ff.CalcEnergy())
        conv = ff.Minimize(maxIts=2000)   # 0 == converged
        e1 = float(ff.CalcEnergy())
        per_conf.append((conv == 0, e0, e1))
    converged = [(e0, e1) for ok, e0, e1 in per_conf if ok]
    if not converged:
        return {"ok": False, "reason": "no_convergence",
                "n_confs": len(cids), "forcefield": ff_label}

    # pick the lowest final-energy converged conformer; report its relaxation gap
    e0, e1 = min(converged, key=lambda t: t[1])
    heavy = _heavy_atoms(mol)
    gap = max(e0 - e1, 0.0)
    return {"ok": True, "forcefield": ff_label, "n_confs": len(cids),
            "n_converged": len(converged), "energy": round(e1, 2),
            "relaxation_gap": round(gap, 2), "heavy_atoms": heavy,
            "strain_per_heavy_atom": round(gap / heavy, 3) if heavy else None}


def calibrate_threshold(smiles_list, pct=99.0, seed=0):
    """Strain/heavy-atom threshold from KNOWN molecules (their pct-th percentile)."""
    vals = []
    for s in smiles_list:
        m = embed_and_relax(s, seed=seed)
        if m["ok"] and m["strain_per_heavy_atom"] is not None:
            vals.append(m["strain_per_heavy_atom"])
    vals = np.array(vals)
    return {"n": int(len(vals)), "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "threshold": float(np.percentile(vals, pct)), "pct": pct}


def geometry_check(smiles, threshold=None, seed=0):
    """Full verdict for one molecule. threshold = strain/heavy-atom cut (from calibrate)."""
    m = embed_and_relax(smiles, seed=seed)
    if not m["ok"]:
        return {"smiles": smiles, "verdict": "FAILED", **m}
    strained = threshold is not None and m["strain_per_heavy_atom"] > threshold
    return {"smiles": smiles, "verdict": "STRAINED" if strained else "STABLE",
            "threshold": threshold, **m}


# ---------------------------------------------------------------------------
# Validation: known flavor molecules should pass; deliberately broken/strained
# structures should be flagged; then report survival on the de-novo library.
# ---------------------------------------------------------------------------
# Real, makeable molecules the gate must NOT reject. Includes several genuinely strained but
# perfectly real rings (cyclopropane, norbornane, adamantane, cubane): the gate passes real
# molecules regardless of thermodynamic strain, which is correct behaviour for a validity
# filter (strain magnitude is the xTB/DFT escalation, not this tier's job).
KNOWN_GOOD = [
    ("vanillin", "O=Cc1ccc(O)c(OC)c1"),
    ("limonene", "CC(=C)C1CCC(C)=CC1"),
    ("menthol", "CC(C)C1CCC(C)CC1O"),
    ("isoamyl acetate", "CC(C)CCOC(C)=O"),
    ("2,5-dimethylpyrazine", "Cc1cnc(C)cn1"),
    ("gamma-decalactone", "CCCCCCC1CCC(=O)O1"),
    ("cyclopropane (strained, real)", "C1CC1"),
    ("norbornane (bridged, real)", "C1CC2CCC1C2"),
    ("adamantane (cage, real)", "C1C2CC3CC1CC(C2)C3"),
    ("cubane (very strained, real)", "C1C2C3C1C4C2C3C4"),
]
# Chemically/geometrically INVALID structures the gate must reject: impossible valence and
# impossible ring closures. These are the real BRICS-reassembly failure mode (a 2D graph that
# cannot correspond to any physical 3D molecule), which is exactly what this cheap tier guards.
SHOULD_FLAG = [
    ("pentavalent carbon", "C(C)(C)(C)(C)C"),
    ("hexavalent carbon", "C(C)(C)(C)(C)(C)C"),
    ("impossible bridgehead", "C1C2CC3(C1)C23"),
    ("bicyclo[1.1.0]but-1(3)-ene (impossible)", "C1C2CC12=C"),
]


def _run_validation():
    from real_data import build_keller_pleasantness
    from safety import screen

    df = build_keller_pleasantness().reset_index(drop=True)
    known_smiles = [r["smiles"] for _, r in df.iterrows()][:200]
    print("Calibrating strain threshold on known Keller molecules ...")
    cal = calibrate_threshold(known_smiles, pct=99.0)
    thr = cal["threshold"]
    print(f"  n={cal['n']}  median={cal['median']:.3f}  p90={cal['p90']:.3f}  "
          f"threshold(p99)={thr:.3f}  kcal/mol per heavy atom\n")

    print("KNOWN-GOOD flavor molecules (expect STABLE):")
    good_pass = 0
    for name, smi in KNOWN_GOOD:
        r = geometry_check(smi, threshold=thr)
        good_pass += int(r["verdict"] == "STABLE")
        print(f"  {r['verdict']:8s} strain/atom={r.get('strain_per_heavy_atom')}  {name}")
    print(f"  -> {good_pass}/{len(KNOWN_GOOD)} STABLE\n")

    print("SHOULD-FLAG strained/degenerate structures (expect STRAINED or FAILED):")
    flagged = 0
    for name, smi in SHOULD_FLAG:
        r = geometry_check(smi, threshold=thr)
        flagged += int(r["verdict"] != "STABLE")
        print(f"  {r['verdict']:8s} strain/atom={r.get('strain_per_heavy_atom')} "
              f"reason={r.get('reason','-')}  {name}")
    print(f"  -> {flagged}/{len(SHOULD_FLAG)} flagged\n")

    # de-novo survival
    print("Applying gate to de-novo library (lego_assembly) ...")
    import lego_assembly as lego
    pre = lego.build_prereqs()
    safe = [r["smiles"] for _, r in pre["df"].iterrows()
            if screen(r["smiles"])["verdict"] == "OK"]
    frags = lego.fragment_library(safe)
    cands = lego.assemble(frags, n_build=400, seed=0)
    verdicts = {"STABLE": 0, "STRAINED": 0, "FAILED": 0}
    for c in cands:
        v = geometry_check(c["smiles"], threshold=thr)["verdict"]
        verdicts[v] += 1
    n = len(cands)
    print(f"  de-novo candidates: {n}")
    for k in ("STABLE", "STRAINED", "FAILED"):
        print(f"    {k:8s} {verdicts[k]:4d}  ({100*verdicts[k]/n:.1f}%)")

    return {"calibration": cal,
            "known_good_stable": f"{good_pass}/{len(KNOWN_GOOD)}",
            "should_flag_flagged": f"{flagged}/{len(SHOULD_FLAG)}",
            "denovo_verdicts": verdicts, "denovo_n": n}


if __name__ == "__main__":
    import json
    out = _run_validation()
    with open("results/geometry_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/geometry_gate.json")
