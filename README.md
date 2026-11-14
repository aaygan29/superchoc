# superchoc

**Can we tell, from molecular structure alone, whether a new combination of flavor molecules
will be liked, before anyone smells it?**

This project is a test of Conjecture 4 in Jake Wintermute's essay *Four Biological
Conjectures* (American Wetware, 2026): "Superchocolate exists in flavor space". The essay
asks for three things: predict the experience of a flavor from structure, generate a new
flavor as delicious as chocolate, and carry the method over to drug design (odor receptors
are GPCRs). The conjecture and framing are the essay's; the experiment and analysis here
are the author's.

The repo does not claim to have found superchocolate. It makes one tested claim and one
pre-committed test:

1. **Tested on independent human data (pre-registered).** A structure-only model that has
   never seen the test molecules predicts how pleasant a trained panel found 198 two-odorant
   mixtures (Ma et al. 2021): Spearman ρ = 0.49 [0.27, 0.65], permutation p = 0.001.
2. **Pre-committed prospective test.** The same frozen model scores new, cocoa-anchored
   mixtures of approved flavorings that no panel has rated. Its predictions are fixed and
   hashed before tasting, with a protocol for a panel to check them.

## Result (pre-registered; `methods/chocolate_anchor/`)

The analysis plan was committed (`61a9465`) and pushed before any structure prediction was
compared with the test ratings. Training data: Keller & Vosshall 2016 single-molecule
pleasantness, with every test molecule removed (44 molecules; zero overlap asserted).

| Hypothesis | Result (95% CI, molecule-cluster bootstrap) | Verdict (Holm) |
|---|---|---|
| H1: predicted vs panel-rated pleasantness of the 72 single odorants | ρ = 0.64 [0.47, 0.76] | supported |
| H2: predicted vs panel-rated pleasantness of 198 mixtures | ρ = 0.49 [0.27, 0.65] | supported |
| H3: model disagreement identifies the mixtures it predicts better | gap 0.02 [-0.20, 0.26] | not supported |

Context (descriptive): molecular weight alone gives ρ = 0.13, so the model is not tracking
size. Using the panel's *measured* component ratings with the same mixture rule gives
ρ = 0.90, so most of the remaining error is in predicting each component, not in combining
them. Full table and every deviation from the plan: `RESULTS_MIXTURE_HEDONICS.md`.

## Post-registration, exploratory (`EXPLORATORY.md`)

Labelled exploratory because they were designed after the plan was fixed.

- **Noise ceiling.** Panel split-half reliability caps any predictor at ρ ≈ 0.94 on these
  mixtures. H2 reaches 52% of the ceiling; measured component ratings reach 96%.
- **Calibrated abstention.** Split-conformal intervals (Angelopoulos & Bates 2021),
  calibrated on molecules disjoint from the test molecules, hit their coverage targets
  (79% at 80%, 89% at 90%). When they commit to liked or disliked they are right 73-77% of
  the time, but they commit on only 6-21% of mixtures.
- **Power.** A replication of Ma's protocol with 40 new mixtures has about 95% power to
  detect the H2 effect; detecting ρ = 0.3 needs about 60.

## Prospective study (`data/candidates/`)

- **Pool.** Approved flavorings only: each has a FEMA GRAS number or an EU Union List
  (Reg. 872/2012) entry, FDA 2018 delistings are excluded, and Ma's molecules are excluded.
  Chocolate's odor-active compounds (OAV ≥ 1 in published dark-chocolate and cocoa
  analyses) anchor every pair. See `POOL.md`.
- **Frozen predictions.** Every cocoa-anchored pair is scored in parallel by the frozen
  model. 40 pairs are selected across the full predicted range, not just the predicted
  favourites, so the panel tests the model rather than confirming its picks. The selection
  and its SHA-256 are in `FROZEN_PREDICTIONS.*`.
- **Protocol.** `methods/chocolate_anchor/PROTOCOL.md`. A Stage 1 Registered Report draft:
  hypotheses, panel procedure replicating Ma 2021, sample size, analysis, and what counts as
  failure.

## What this does not show

- Nothing has been tasted for this project. "Liked" means predicted panel pleasantness of
  an odor mixture, in the sweet/food-odorant range Ma sampled, by a European trained panel.
  Person-to-person variation is large (Arshamian et al. 2022), so a mixture can be liked on
  average and disliked by many people.
- The evidence covers binary mixtures at roughly matched intensity. Real chocolate has
  dozens of odor-active compounds at specific concentrations. Extending to more components
  and to concentration needs data that does not yet exist publicly.
- Regulatory status covers use as a flavoring at customary levels. It is not a safety
  clearance for any particular stimulus; a panel study needs ethics approval and customary
  dilutions.
- The receptor-level and drug-design parts of the conjecture are not addressed. The
  receptor model kept in `methods/real_flavor/m2or_receptor.py` is exploratory and enters
  no claim.

## Repository map

```
methods/chocolate_anchor/   the experiment: plan, registered analysis, exploratory analyses,
                            discrimination model (Bushdid 2014), candidate pool, frozen predictions
methods/real_flavor/        reused v1 components: Keller loader, featurization, pleasantness
                            model, Snitz/Ravia mixture distance, Leffingwell profiles, hazard filter
data/hedonics/              Ma 2021 mixture ratings and other openly licensed rating data
data/sensomics/             cocoa key-odorant concentrations, thresholds, OAVs, with sources
data/candidates/            candidate pool, regulatory evidence, frozen predictions
docs/literature/            verified references, retraction checks, methods integrated
docs/REMOVED.md             every v1 module removed and why (all still at tag v1-archive)
```

## Reproduce

```bash
uv venv && uv pip install -r requirements.txt
export OMP_NUM_THREADS=1 PYTHONPATH=methods
python -m pytest -q methods
python -m chocolate_anchor.mixture_hedonics     # registered analysis, ~16 min
python -m chocolate_anchor.exploratory          # ceiling, conformal, power
python -m chocolate_anchor.prospective          # scores pairs, writes frozen predictions
```

## History

Version 1 built a broad pipeline (hyperbolic flavor geometry, receptor layers, de-novo
molecules, recipe classifiers). Its own self-audit and a later module-by-module audit found
most of those claims unsupported, so they were removed: see `docs/REMOVED.md`, and tag
`v1-archive` to run them.
