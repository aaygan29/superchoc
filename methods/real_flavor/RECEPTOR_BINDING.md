# Structure-based receptor-interaction layer (Boltz-2 co-folding)

Wintermute's chain is molecule -> receptor activation -> percept. In this repo `receptors.py`
grounds the middle only through a sparse *structural nearest-neighbour* to Mainland 2015
confirmed OR-ligand pairs (a ~1.6x-chance signal that never looks at the receptor's structure).
`receptor_binding.py` upgrades that to a genuine **structure-based** interaction score: it
co-folds a flavor molecule into a real odorant/taste receptor with Boltz-2 (api.boltz.bio) and
reads out the complex binding confidence.

This is zero-shot with respect to this repo: Boltz has never seen the Keller/Mainland labels,
so a molecule that is in no training set here can still be scored against a receptor.

## Receptors (real human sequences, UniProt, retrieved 2026-09-11)

| receptor | UniProt | cognate class | reference |
|---|---|---|---|
| OR5AN1 | Q8NGI8 | macrocyclic musks (muscone) | Shirasu et al. 2014, *Neuron* |
| OR51E2 | Q9H255 | short-chain fatty acids (propionate/acetate) | Saito/Fujita |

## Validation design (`validate_receptor_binding.py`) — a specificity swap test

For each receptor, its **cognate** ligand class should co-fold with higher binding confidence
than the **non-cognate** class:

```
OR5AN1: musks (muscone, civetone)      >  acids (propionic, acetic)
OR51E2: acids (propionic, acetic)      >  musks (muscone, civetone)
```

These molecule->receptor assignments come from external pharmacology, **not** from the
Keller/Mainland labels used elsewhere here. A correct cognate > non-cognate ranking is therefore
a genuine out-of-training-set prediction of a known pairing, which is exactly the claim we want
to support or refute. The script reports per-pair scores, a per-receptor cognate-vs-non-cognate
margin, and a separation AUC, and writes `results/receptor_binding.json`.

## Honesty

Odorant receptors are 7-transmembrane GPCRs and Boltz-2's binding head is trained mostly on
soluble protein-ligand complexes, so it is **not assumed to work** on this class. The validation
is adversarial: if the co-fold does not separate cognate from non-cognate ligands, that is the
reported finding and the structural-NN layer in `receptors.py` remains the receptor grounding.
If it does separate, it becomes a real structure-based upgrade to the weakest quantitative link
in the pipeline.

## Result (8 co-folds, `results/receptor_binding.json`)

The score is `ligand_iptm` (the ligand interface pTM from Boltz-2); the runs did not emit a
separate affinity head, so this is a pose-confidence proxy, not a measured affinity.

| receptor | cognate class mean | non-cognate class mean | margin | direction |
|---|---|---|---|---|
| OR5AN1 (musk) | 0.940 | 0.729 | **+0.211** | correct |
| OR51E2 (acid) | 0.838 | 0.793 | **+0.045** | correct |

Per-pair `ligand_iptm`: OR5AN1 muscone 0.961, civetone 0.919, acetic 0.833, propionic 0.625;
OR51E2 propionic 0.891, muscone 0.816, acetic 0.786, civetone 0.771.

**Separation AUC (cognate > non-cognate) = 0.875, 2/2 receptors in the correct direction ->
verdict SEPARATES.** The structure-based co-fold recovers receptor specificity for molecule
-receptor pairs that are not in this repo's training data.

Honest caveats (do not overclaim):
- n is small (2 receptors x 4 ligands). This is a proof of concept, not a benchmark.
- OR51E2's margin is thin (+0.045): muscone scores 0.816 on the acid receptor, above cognate
  acetic acid (0.786). Discrimination is much stronger for OR5AN1 than OR51E2.
- `ligand_iptm` is interface-pose confidence, which correlates with but is not binding affinity.
- Boltz-2 on 7TM GPCRs is out-of-distribution; that it separates here is encouraging, not proof
  the score is calibrated to real EC50s.

## Reproduce

```bash
python validate_receptor_binding.py --estimate   # confirm cost (~$0.05 each), no GPU
python validate_receptor_binding.py               # run/reload the 8 co-folds, write results
```

Results cache under `results/boltz/`, so re-running is free after the first pass.

## Reproduce a single co-fold

```bash
python -c "import receptor_binding as rb, json; print(json.dumps(rb.cofold('OR51E2','CCC(=O)O')['metrics'], indent=2))"
```
Requires `boltz-api` (installed) and an OAuth session (`boltz-api auth login`).
