# real_flavor: components reused from version 1

What survived the v1 audit (see `../../docs/REMOVED.md` for what did not).

| file | what it does | used by |
|---|---|---|
| `real_data.py` | Keller & Vosshall 2016 single-molecule pleasantness (pyrfume), 1/1000 dilution | `chocolate_anchor` training data |
| `featurize.py` | SMILES to ECFP4, Tanimoto | featurization |
| `goodness_real.py` | v1 random forest on ECFP4. Random 5-fold Spearman 0.50; scaffold split 0.30 (the extrapolation figure) | reference only |
| `mixtures.py` | angle distance between mixtures (Snitz 2013); tracks human similarity, Snitz rho -0.49, Ravia rho -0.25. Similarity, not pleasantness | `chocolate_anchor.discrimination` |
| `flavor_profile.py` | Leffingwell odor descriptors (pyrfume) | `chocolate_anchor.discrimination` |
| `significance.py` | bootstrap / permutation / FDR for the Snitz and Ravia results | `results/significance.json` |
| `m2or_receptor.py` | molecule-to-odorant-receptor model on M2OR, leave-molecule-out AUC 0.89 (pooled). Exploratory; no pleasantness claim | none |
| `safety.py` | structural hazard filter. Not a food-safety or regulatory clearance; eligibility comes from the FEMA / EU lists in `data/candidates/` | candidate pool |
| `reference_flavors.py` | literature key-odorant sets for 23 flavors; non-cocoa entries are loosely sourced | reference only |

`HANDOFF.md` is the checklist anyone must follow before smelling or tasting anything.
`results/NEWSLATE_findings.md` is the v1 self-audit, kept as the record of what was tested.
