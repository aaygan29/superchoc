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

## Status

Layer and validation harness are built and schema-verified against the live API (the top-level
`binding: {type: ligand_protein_binding, binder_chain_id: B}` block validates; each co-fold is
~$0.05, so the 8-run matrix is ~$0.40). **Runs are currently blocked on Boltz account credits**
(the API returned `402 payment_required`, "All credits depleted"). Add prepaid credits or enable
on-demand billing at the api.boltz.bio console, then:

```bash
python validate_receptor_binding.py --estimate   # confirm cost, no GPU
python validate_receptor_binding.py               # run the 8 co-folds, cache, write results
```

Results cache under `results/boltz/`, so re-running is free after the first pass.

## Reproduce a single co-fold

```bash
python -c "import receptor_binding as rb, json; print(json.dumps(rb.cofold('OR51E2','CCC(=O)O')['metrics'], indent=2))"
```
Requires `boltz-api` (installed) and an OAuth session (`boltz-api auth login`).
