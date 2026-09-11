# Geometry gate: is a de-novo molecule a physically realizable 3D structure?

`lego_assembly.py` snaps BRICS fragments into new molecules and keeps whatever RDKit can
sanitize as a 2D valence graph. A valid 2D graph is not the same as a molecule that can adopt a
real 3D geometry: fragment reassembly can in principle emit impossible valences, impossible ring
closures, or non-embeddable cages. `geometry_gate.py` is a cheap 3D-validity filter that runs
before a candidate is ever proposed as a flavor building block.

## What it computes (`geometry_gate.py`)

1. Parse, add hydrogens.
2. Embed several conformers with ETKDGv3 (distance geometry).
3. Relax each with a molecular-mechanics force field (MMFF94 where parameterized, else UFF).
4. Strain = the **relaxation gap** (single-point energy of the raw embedding minus the
   minimized energy) per heavy atom. This is a difference of energies for the *same* graph, so
   the composition-dependent baseline cancels. (Absolute FF energy per atom does not cancel it,
   which is why an earlier version false-flagged aromatic heterocycles like pyrazine.)

Verdict: `FAILED` (cannot parse / embed / converge), `STRAINED` (relaxation-gap outlier vs the
known-flavor threshold; a soft flag), or `STABLE`.

## What it does and does NOT do

- **DOES** reject chemically/geometrically invalid structures (impossible valence, impossible
  ring closure) and **pass every real molecule, including genuinely strained-but-makeable
  rings**. This is the correct behaviour for a validity filter.
- **Does NOT** rank thermodynamic ring strain. MMFF/UFF embed small rigid strained cages
  cleanly, so cubane looks as stable as cyclohexane. Ranking strain magnitude needs
  semiempirical xTB or DFT, which drops in behind the same `geometry_check -> dict` interface.
  That is the deliberate escalation for the top few candidates, not a claim this cheap tier
  makes.

## Validation (`python geometry_gate.py`, artifact `results/geometry_gate.json`)

| test set | result |
|---|---|
| Known-good flavor molecules (incl. cyclopropane, norbornane, adamantane, cubane) | **10/10 STABLE** (no real molecule rejected) |
| Chemically-invalid structures (penta/hexavalent carbon, impossible bridgeheads) | **4/4 FAILED** (all rejected) |
| De-novo library (BRICS reassembly) | **100% STABLE** (a QA pass: reassembly is geometrically sound) |

The known-good set includes strained-but-real rings on purpose: passing them is the point. The
should-flag set is the genuine fragment-reassembly failure mode (a 2D graph that maps to no
physical molecule), which is exactly what the gate guards.

## How it plugs into the pipeline

`lego_assembly.run()` now calibrates a strain threshold once on a sample of known safe
molecules, tags every safe/novel candidate with its geometry verdict + strain, and **drops any
non-`STABLE` candidate** before the PubChem novelty check. Result keys `n_geometry_stable` and
`geometry_threshold` record the effect. Escalating the top survivors to xTB/DFT (real strain and
conformer energetics) is the natural next tier.

## Reproduce

```bash
python geometry_gate.py     # calibrate + known-good + invalid + de-novo survival
```
Dependencies: RDKit only (no external quantum-chemistry binary for this tier).
