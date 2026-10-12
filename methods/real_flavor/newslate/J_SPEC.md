# J specification (what the code actually does)

Machine-readable twin: `configs/J_spec.json` (read by `jspec.py`). Implementation: `jspec.py`. Equivalence test against the legacy generator: `test_jspec.py`.

## Does the code match the paper?
The paper never defined J, the initialisation, the optimiser, the stopping rule or the hyperparameters. The legacy generator (`denovo_recipes.compose`) implements an
**additive** objective `J = 1.0*T + 0.15*C + 0.4*N` and never calls the non-additive mixture model, so the paper's motivation (an additive blend cannot beat its best
part) was not acted on. Two further legacy facts: (1) taste for known molecules is the in-sample prediction of a model trained on those very molecules; (2) the
initial recipe may silently violate the novelty constraint if 100 resamples fail (counted as `infeasible_init`; 0 occurrences in the NEWSLATE runs).

## Objective
`J(R) = w_taste*T(R) + w_comp*C(R) + w_novel*N(R) + w_mix*M(R)`, R a set of 10 distinct molecules at equal proportions. Base weights 1.0, 0.15, 0.4, 0.4.

| term | formula | range in practice |
|---|---|---|
| T | mean over components of predicted pleasantness / 100. RF(300 trees, seed 0) on ECFP4 (r=2, 2048 bits), Keller 2016 (n=420). Known molecules use 5-fold out-of-fold predictions (`oof`); de novo molecules use the full-data model. | about 0.5 to 0.7 |
| C | mean over components of (#favoured classes - #avoided classes), 15 SMARTS classes, favour/avoid = top-5 positive / bottom-5 negative class lift over all 420 labels | about -1 to 4 |
| N | min over the 6 reference flavors f of (1 - mean_i max_r Tanimoto(i, r in f)). Hard constraint N >= 0.5 | 0.5 to 0.8 |
| M | (100 - max_f s_hat(R, f)) / 100, s_hat = set-pooled RF (300 trees, min_samples_leaf 2) trained on all 555 Snitz+Ravia human similarity pairs, features [angle(mean), abs diff of mean/std/min/max] over 13 z-scored RDKit descriptors. Whole-blend, non-additive in the components. | about 0.5 to 0.7 |

M is "predicted perceptual distinctiveness of the blend from every reference flavor". The human data it is trained on are similarity ratings, not pleasantness: no
human mixture-pleasantness labels exist in the repo's datasets, so the non-additive term cannot claim to model goodness.

## Search
- Pool: 420 safety-OK known molecules + first 40 safe de novo candidates of `results/lego_denovo.json`.
- Init: per restart draw 10 distinct eligible molecules uniformly, resample up to 100x until N >= 0.5.
- Optimiser: stochastic first-improvement 1-swap hill climbing. Proposal: uniform random eligible molecule v and slot k, rejected if v already present. Accept iff N >= 0.5 and J strictly increases.
- Stopping rule: fixed budget, 500 proposals per restart, 5 restarts, no early stopping, no temperature. Restarts that converge to the same set are collapsed. Sorted by J.
- RNG: `numpy.random.default_rng(seed)`, per proposal `choice(eligible)` then `integers(10)`.
- Proportions are equal during search; Scheffe ratio optimisation is a separate post-hoc step on the top recipe and is not part of J.
