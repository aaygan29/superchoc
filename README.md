# superchoc

**Can molecular structure alone predict how pleasant people will find the smell of a new
combination of flavor molecules, before anyone smells it?**

This project is a test of Conjecture 4 in Jake Wintermute's essay *Four Biological
Conjectures* (American Wetware, 2026): "Superchocolate exists in flavor space". The essay
asks for three things: predict the experience of a flavor from structure, generate a new
flavor as delicious as chocolate, and carry the method over to drug design (odor receptors
are GPCRs). The conjecture and framing are the essay's; the experiment and analysis here
are the author's.

The repo does not claim to have found superchocolate. It makes one tested claim and one
pre-committed test:

1. **Tested on independent human data, with the analysis plan fixed in advance.** A
   structure-only model whose training data excludes every test molecule predicts how
   pleasant a trained panel found the smell of 198 two-odorant mixtures (Ma et al. 2021):
   Spearman ρ = 0.49 [0.27, 0.65], permutation p = 0.001.
2. **A fixed prospective test.** The same code, retrained without the new molecules,
   scores new cocoa-anchored mixtures of listed flavorings that no panel has rated. Its
   predictions are fixed and hashed before any rating, with a protocol for a panel to
   check them.

## Result (pre-registered; `methods/chocolate_anchor/`)

**How the plan was fixed.** The analysis plan was committed as `61a9465`. GitHub's
activity log shows it on the remote from 01:29:49 UTC on 2026-11-14, before any analysis
output existed. The analysis code was written after that and is not part of the
registration; how it interprets the plan is listed in the results file. There is no
independent registry entry for this retrospective analysis.

Training data is Keller & Vosshall 2016 single-molecule pleasantness. Every test molecule
was removed (44 molecules, matched by PubChem CID and by structure, with zero overlap
asserted). Close relatives remain: 10 test molecules have an ECFP4 Tanimoto similarity of
at least 0.9 to a training molecule. Restricting to mixtures without them gives ρ = 0.51
[0.27, 0.67].

| Hypothesis | Result (95% CI, molecule-cluster bootstrap) | Verdict (Holm) |
|---|---|---|
| H1: predicted vs panel-rated pleasantness of the 72 single odorants | ρ = 0.64 [0.47, 0.76] | supported |
| H2: predicted vs panel-rated pleasantness of 198 mixtures | ρ = 0.49 [0.27, 0.65] | supported |
| H3: model disagreement identifies the mixtures it predicts better | gap 0.02 [-0.20, 0.26] | criteria met as registered; not supported under a stricter test added afterwards (see below) |

**H3.** Both registered criteria are met, so under the plan's wording H3 is supported. The
analysis then added a p-value for the multiple-comparison correction, defined after
registration, and under it H3 is not supported. The effect is a 0.02 difference in AUROC,
so we make no claim that abstaining on model disagreement helps.

**How much of this is just simple chemistry.** Molecular weight alone gives ρ = 0.13. A
simple model fitted to 8 basic descriptors (size, polarity, hydrogen bonding, rings,
sulfur) gives ρ = 0.39 [0.10, 0.60]. The full model's advantage over it is +0.10
[-0.03, 0.25], which is not statistically clear. So structure predicts mixture
pleasantness, but this data cannot show that the four-model ensemble does better than a
simple descriptor model. All baseline and robustness analyses were done after registration
(`ROBUSTNESS.md`).

**Where the error comes from.** Using the panel's *measured* component ratings with the
same mixture rule gives ρ = 0.90. Most of the remaining error is therefore in predicting
each component, not in combining them. Ma's mixtures span a narrow range (panel means
about 2 to 6.8 on a 0-10 scale), so correlations on a wider range of odors may differ.

The full table, and every way the code interprets the plan, is in
`RESULTS_MIXTURE_HEDONICS.md` and the dated note in `PREREGISTRATION.md`.

## Post-registration, exploratory (`EXPLORATORY.md`)

Labelled exploratory because they were designed after the plan was fixed.

- **Noise ceiling.** Panel split-half reliability caps any predictor at ρ ≈ 0.94 on these
  mixtures. H2 reaches 52% of the ceiling; measured component ratings reach 96%.
- **Calibrated abstention.** Split-conformal intervals (Angelopoulos & Bates 2021),
  calibrated on molecules disjoint from the test molecules, hit their coverage targets
  (79% at 80%, 89% at 90%). When they commit to liked or disliked they are right 73-77% of
  the time, but they commit on only 6-21% of mixtures.
- **Power.** For the prospective design actually fixed (40 pairs that share molecules,
  tested with the combined rule below), simulated power is 0.74 if the true correlation is
  0.49 and 0.31 if it is 0.3. The false-positive rate is 0.019 (`ROBUSTNESS.md`).

## Prospective study (`data/candidates/`)

- **Pool.** Listed flavorings only. Each molecule has a FEMA GRAS number or an entry in the
  EU Union List as adopted in Reg. 872/2012 (later amendments were not checked). FDA 2018
  delistings are excluded, and so are Ma's molecules.
  Chocolate's odor-active compounds (OAV ≥ 1 in published dark-chocolate and cocoa
  analyses) anchor every pair. See `POOL.md`.
- **Frozen predictions.** All 693 cocoa-anchored pairs are scored in parallel shards.
  The model is the same code retrained without the test molecules; on Ma it scores
  ρ = 0.46, and its predictions correlate 0.96 with the registered model's. 40 pairs are
  chosen across the whole predicted range, not just the predicted favourites, so the panel
  tests the model instead of confirming its picks. The pair and molecule predictions, and
  their SHA-256 values, are in `FROZEN_*`. Version 1 of this selection was replaced after
  the review, before anything was rated.
- **Protocol.** `methods/chocolate_anchor/PROTOCOL.md` is a draft for a Stage 1 Registered
  Report that no journal has reviewed yet. It sets out the hypotheses, a panel procedure
  following Ma 2021 (lab details marked for the running lab), the analysis fixed in
  `stage2_analysis.py`, and what counts as failure.

## What this does not show

- Nothing has been smelled or tasted for this project. Every claim is about the average
  pleasantness a trained European panel gives a sniffed odor mixture, in the food-odorant
  range Ma sampled. It is not about taste, retronasal flavor, or whether a food is
  "delicious".
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
methods/chocolate_anchor/   the experiment: plan, registered analysis, exploratory and robustness
                            analyses, candidate pool, frozen predictions, protocol, Stage 2 code.
                            Also an exploratory triangle-test model on Bushdid 2014; that dataset's
                            headline "1 trillion smells" claim is disputed and is not used
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
python -m chocolate_anchor.robustness           # baselines, near-duplicates, P1 rule power
python -m chocolate_anchor.prospective          # scores pairs, writes frozen predictions
```

## History

Version 1 built a broad pipeline (hyperbolic flavor geometry, receptor layers, de-novo
molecules, recipe classifiers). Its own self-audit and a later module-by-module audit found
most of those claims unsupported, so they were removed: see `docs/REMOVED.md`, and tag
`v1-archive` to run them.
