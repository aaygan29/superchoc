# Molecular flavor literature: synthesis and technique library

A working synthesis of the molecular-flavor and olfaction literature, organized so it
doubles as a library of experimental techniques and computational tools to draw from.
Every load-bearing claim is tied to an external, already-published source. Attribution
note: entries retrieved through PubMed are cited with their DOI, per PubMed terms of use.

The through-line for this repo: **molecule structure to receptor activation to perceived
flavor**, learned against human report, with the olfactory-receptor half of the problem
sharing GPCR machinery with drug design (see [Criterion 3](../ROADMAP.md)).

---

## 1. What chocolate flavor actually is (the target)

Chocolate is not one molecule. It is a high-dimensional mixture, and the field measures
it with a specific, reusable toolchain rather than a single assay.

- **Cocoa/chocolate flavor is a mixture of hundreds of volatiles, but only dozens are
  odor-active.** Reviews of cocoa-liquor flavor chemistry consolidate the sensory and
  instrumental picture and argue for pairing trained-panel profiling with instrumental
  analysis. Chaudhary, Mongia & Drake, 2025, *J Food Sci*.
  [DOI](https://doi.org/10.1111/1750-3841.70481)
- **The discriminative key odorants of dark chocolate are identifiable and countable.**
  A GC-olfactometry study with 12 assessors isolated 38 discriminant key odorants (19
  most discriminant), dominated by heterocyclics (furanones, pyranones, lactones,
  pyrrole, pyrazine). This is the empirical basis for "chocolate needs dozens of
  molecules, not one." Deuscher et al., 2020, *Molecules*.
  [DOI](https://doi.org/10.3390/molecules25081809)
- **Chocolate-like aroma can be produced and reverse-engineered from a small key set.**
  A fermentation route reproduced chocolate-like aroma; 15 key aroma compounds were
  identified and confirmed by a recombination study, with dihydroactinidiolide,
  isovaleraldehyde and coumarin dominant by odor activity value (OAV). Recombination +
  OAV is the standard "did we capture the essence?" test. Rigling et al., 2022,
  *Molecules*. [DOI](https://doi.org/10.3390/molecules27082503)

**Reusable techniques from this cluster (the wet-lab side):**

| Technique | What it gives us | Role in this repo |
|---|---|---|
| GC-MS / GC-Olfactometry (GC-O) | Which volatiles are present *and* which are smelled | Ground-truth labeling of mixtures |
| Odor Activity Value (OAV) | Concentration / odor threshold ranking | Prioritize which molecules matter |
| Aroma recombination + omission | Test whether a molecule subset reproduces the percept | Validation of a "minimal chocolate" |
| Trained sensory panel / detection-frequency analysis | Human descriptors, the real target variable | The label the whole model is fit against |
| Stir-bar sorptive extraction / headspace | Sampling volatiles reproducibly | Front of the data pipeline |

---

## 2. Structure to percept (the forward model, Criterion 1)

The central computational result of the last few years: a learned embedding of molecules
predicts human odor better than hand-built chemistry features, and the embedding
generalizes.

- **The Principal Odor Map (POM).** A graph neural network trained on expert odor labels
  produces an embedding that, on 400 prospective novel odorants, matched the trained
  panel mean better than the median panelist did (human-level), and transferred to
  intensity and perceptual-similarity tasks it was never trained on. This is the
  strongest existing evidence that flavor space is learnable and continuous. Lee et al.,
  2023, *Science* 381:999-1006.
  [science.org](https://www.science.org/doi/10.1126/science.ade4401) /
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/37651511/)
- **The POM is not just perceptual, it tracks biology.** Dissecting the model showed POM
  representations predict odor-evoked receptor, neural and behavioral responses across
  many species, and that metabolically co-occurring odorants sit close in the map. This
  is why the embedding is a defensible "ground truth" coordinate system rather than a
  fit artifact. Qian et al., 2023, *eLife* 12:e82502.
  [DOI](https://doi.org/10.7554/eLife.82502)
- **Personalized, multi-attribute prediction is feasible from chemoinformatic features.**
  The DREAM Olfaction Prediction Challenge winning model (random forest) predicted
  individual and population perceptual ratings, and found a small set of low-degeneracy
  features carries most of the signal. Establishes the baseline model class and the
  inter-individual variability problem. Li et al., 2018, *GigaScience* 7:1-11.
  [DOI](https://doi.org/10.1093/gigascience/gix127)

**Reusable computational tools/methods from this cluster:**

- Graph neural networks over molecular graphs -> learned odor embedding (POM approach).
- Random forests / gradient boosting over chemoinformatic descriptors (DREAM baseline,
  strong, interpretable, cheap first model).
- SHAP feature attribution to find which substructures drive a percept (see saltiness
  study below) -> interpretable structure-odor rules.
- Treat odor labeling as **multi-label** classification (a molecule has many descriptors).

---

## 3. Receptor-level mechanism (the bridge to Criterion 3)

Odor receptors are GPCRs, so the same modeling that explains flavor is the machinery of
GPCR drug design. This is the repo's core transfer bet.

- **Computational modeling of olfactory receptors is now a mature toolbox.** A 2025
  review lays out homology modeling, docking, molecular dynamics, free-energy
  calculations, pharmacophore modeling, virtual screening, and ML prediction for ORs,
  and notes AlphaFold has shifted the field toward structure-based approaches while
  ligand-based methods remain useful when structures are uncertain. This is the method
  menu for the receptor layer. Odoemelam, Steuber & Schmuker, 2025, *BBA Gen Subj*
  1869:130825. [DOI](https://doi.org/10.1016/j.bbagen.2025.130825)
- **Structure + ML jointly explain which odorant hits which receptor.** A saltiness-
  enhancement study combined XGBoost (R=0.96) with molecular docking and site-directed
  mutagenesis to localize odorant binding to TM3/TM5/TM6 of OR1A1 and OR1D2 and name key
  residues. A concrete template for the structure -> receptor-activation step, including
  the mechanistic validation. Ji et al., 2025, *Food Res Int* 202:115707.
  [DOI](https://doi.org/10.1016/j.foodres.2025.115707)

---

## 4. GPCR drug design (Criterion 3)

- **Deep-learning structure prediction has materially improved GPCR docking/screening.**
  Across 70 diverse GPCR complexes, docking onto DL-predicted model structures approached
  the success rate of cross-docking on experimental structures, a >30% gain over the best
  pre-DL protocols, conditional on correct functional-state modeling and receptor-flexible
  docking. Sets best-practice and the benchmark bar. Lee et al., 2022, *Comput Struct
  Biotechnol J* 21:158-167. [DOI](https://doi.org/10.1016/j.csbj.2022.11.057)
- **Ligand-based deep learning can find GPCR agonists from tiny training sets.** A DNN
  identified a ~500 nM mu-opioid receptor agonist from a 63-compound training set,
  outperforming classical QSAR/LBVS. Shows a viable path when structural data is thin.
  Tsou et al., 2020, *Sci Rep* 10:16771. [DOI](https://doi.org/10.1038/s41598-020-73681-1)

**Reusable computational tools/methods from clusters 3-4:**

- AlphaFold2 / state-specific GPCR models as docking targets (functional-state matters).
- Receptor-flexible docking + molecular dynamics for pocket realism.
- Ligand-based DNN/QSAR virtual screening when structures are missing.
- Pharmacophore modeling (ligand- and structure-based) for hit expansion.
- SHAP / feature attribution for structure-activity interpretability.

---

## 5. Cross-cutting reading

- Frauendorfer & Schieberle, 2006 (selected aroma compounds in cocoa powder) and Mishra
  et al., 2024 (cacao anatomy and processing) are the source figures cited in the
  conjecture; use them for the canonical cocoa key-odorant list and processing context.
- GPCRs are ~36% of approved-drug targets, which is why solving the OR-GPCR forward model
  plausibly generalizes to pharmacology. [Nature Reviews Drug Discovery,
  2025](https://www.nature.com/articles/s41573-025-01139-y).

See [GROUND_TRUTH.md](GROUND_TRUTH.md) for the concrete datasets and benchmarks that turn
the methods above into something we can fit and measure.

## Sources

- [Chaudhary et al., 2025, J Food Sci](https://doi.org/10.1111/1750-3841.70481)
- [Deuscher et al., 2020, Molecules](https://doi.org/10.3390/molecules25081809)
- [Rigling et al., 2022, Molecules](https://doi.org/10.3390/molecules27082503)
- [Lee et al., 2023, Science (POM)](https://www.science.org/doi/10.1126/science.ade4401)
- [Qian et al., 2023, eLife](https://doi.org/10.7554/eLife.82502)
- [Li et al., 2018, GigaScience (DREAM)](https://doi.org/10.1093/gigascience/gix127)
- [Odoemelam et al., 2025, BBA Gen Subj](https://doi.org/10.1016/j.bbagen.2025.130825)
- [Ji et al., 2025, Food Res Int](https://doi.org/10.1016/j.foodres.2025.115707)
- [Lee et al., 2022, Comput Struct Biotechnol J](https://doi.org/10.1016/j.csbj.2022.11.057)
- [Tsou et al., 2020, Sci Rep](https://doi.org/10.1038/s41598-020-73681-1)

_Biomedical citations above were retrieved via PubMed._
