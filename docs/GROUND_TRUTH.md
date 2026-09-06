# Ground truth: datasets and benchmarks

The point of this file is discipline. "Predict flavor," "as delicious as chocolate," and
"world-beating GPCR model" are only meaningful against fixed, external, public ground
truth. This lists the datasets that anchor each success criterion, what they contain, and
what they can and cannot certify. Nothing here is a result yet. These are the yardsticks.

Rule for the repo: a claim is only allowed once it is measured on one of these held-out
sets against a pre-registered baseline. See [LITERATURE.md](LITERATURE.md) for the
methods that fit against them.

---

## Layer A: structure -> odor percept (Criterion 1 ground truth)

The human-report label is the real target. These are the openly available structure/odor
corpora.

| Dataset | Contents | Why it is ground truth | Access |
|---|---|---|---|
| **Pyrfume** | Dozens of stimulus-linked olfactory datasets unified under one schema (Arctander, AromaDb, FlavorDB, FlavorNet, TGSC/GoodScents, IFRA, Leffingwell, Sigma catalog, etc.) | Single largest curated, machine-readable bridge from molecule to human odor descriptor | [Nature Sci Data 2024](https://www.nature.com/articles/s41597-024-04051-z), `pyrfume-data` on GitHub |
| **GoodScents + Leffingwell** | ~5,030 expert-labeled molecules (3,786 + 3,561, 2,317 overlap), multi-label odor descriptors | Expert descriptor labels; the training set behind the POM and most SOR models | [Leffingwell (Zenodo)](https://doi.org/10.5281/zenodo.4085097) |
| **DREAM Olfaction Challenge (Keller et al.)** | 480 structurally diverse molecules, continuous sensory attributes (intensity, pleasantness) + per-individual ratings | Gold standard for *quantitative* and *personalized* percept prediction; a real held-out benchmark with a known baseline | Published challenge data (Keller et al., 2017, *Science*) |
| **Cocoa/chocolate GC-O key-odorant tables** | Curated key-odorant lists with OAVs for cocoa/dark chocolate | Domain-specific anchor for the chocolate target itself | Deuscher et al. 2020 [DOI](https://doi.org/10.3390/molecules25081809); Rigling et al. 2022 [DOI](https://doi.org/10.3390/molecules27082503) |

**What Layer A can certify:** does structure predict human descriptor / intensity /
similarity on held-out molecules.
**What it cannot:** mixtures at chocolate-scale complexity are underrepresented; most
labels are single-molecule. Mixture percepts must be treated as an open modeling problem
(the DREAM Mixtures track is the closest mixture-distance ground truth).

---

## Layer B: molecule -> receptor activation (mechanistic ground truth, bridges C1 and C3)

| Dataset | Contents | Why it is ground truth | Access |
|---|---|---|---|
| **GPCRdb (2025)** | Reference GPCR sequences, AlphaFold2-MultiState models, integrated ligand data; the 2025 release **adds odorant receptors** | The single resource that unifies the flavor (OR) and drug (non-olfactory GPCR) sides of the repo | [GPCRdb 2025, NAR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11701689/) |
| **Odorant-receptor deorphanization assays** | Experimentally measured OR-odorant activation (in vitro) | Direct labels for the structure -> receptor-activation step | Aggregated in the OR modeling review, Odoemelam et al. 2025 [DOI](https://doi.org/10.1016/j.bbagen.2025.130825) |

**What Layer B can certify:** does a model predict which receptor an odorant activates.
**What it cannot:** OR deorphanization coverage is sparse (most human ORs remain
orphans); absence of a label is not absence of activation.

---

## Layer C: ligand -> GPCR affinity/activity (Criterion 3 ground truth)

| Dataset | Contents | Why it is ground truth | Access |
|---|---|---|---|
| **ChEMBL** | Curated bioactivities (IC50/EC50/Ki) for compounds against targets, incl. GPCRs, with confidence scores | The standard held-out affinity benchmark; confidence score 9 = highest-quality single-protein data | ChEMBL v34 (queryable in this repo's toolchain) |
| **BindingDB** | Measured binding affinities, protein-ligand | Independent affinity ground truth for cross-checking ChEMBL | bindingdb.org |
| **GtoPdb / PDSP Ki / GLASS** | Curated GPCR-ligand associations | Pharmacology-grade GPCR ligand labels; integrated by GPCRdb | [GLASS, Bioinformatics](https://academic.oup.com/bioinformatics/article/31/18/3035/240608) |
| **Experimental GPCR structures (PDB) + the 70-complex docking benchmark** | Resolved GPCR-ligand complexes | Ground truth for docking pose accuracy (RMSD) and virtual-screening enrichment | Lee et al. 2022 [DOI](https://doi.org/10.1016/j.csbj.2022.11.057) |

**What Layer C can certify:** does the model predict binding affinity / correct pose /
screening enrichment on held-out targets and compounds, against published baselines.
**What it cannot:** in-vitro affinity is not clinical efficacy; ADMET and selectivity are
separate axes (use ChEMBL ADMET + off-target panels).

---

## The evaluation contract

1. **Splits are fixed before modeling.** Scaffold/temporal splits for molecules, target-
   holdout splits for GPCRs. No tuning on the test set.
2. **Baselines are pre-registered.** Layer A: random forest on chemoinformatic features
   (DREAM baseline). Layer C: best pre-DL docking + a strong published DL baseline.
3. **Human report is the final arbiter for flavor.** In-silico deliciousness is a proxy;
   the Criterion-2 claim requires a blinded human panel, exactly as in the recombination
   and detection-frequency methods in the flavor-chemistry literature.
4. **Report rescue-or-refute honestly.** A control that refutes a claim ships too.

## Sources

- [Pyrfume, Nature Scientific Data 2024](https://www.nature.com/articles/s41597-024-04051-z)
- [Leffingwell Odor Dataset (Zenodo)](https://doi.org/10.5281/zenodo.4085097)
- [Lee et al., 2023, Science (POM)](https://www.science.org/doi/10.1126/science.ade4401)
- [Qian et al., 2023, eLife](https://doi.org/10.7554/eLife.82502)
- [Li et al., 2018, GigaScience (DREAM)](https://doi.org/10.1093/gigascience/gix127)
- [GPCRdb 2025, NAR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11701689/)
- [Odoemelam et al., 2025, BBA Gen Subj](https://doi.org/10.1016/j.bbagen.2025.130825)
- [GLASS, Bioinformatics](https://academic.oup.com/bioinformatics/article/31/18/3035/240608)
- [Lee et al., 2022, Comput Struct Biotechnol J](https://doi.org/10.1016/j.csbj.2022.11.057)

_Biomedical citations above were retrieved via PubMed and named public databases._
