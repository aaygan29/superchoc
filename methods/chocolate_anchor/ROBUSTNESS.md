# Robustness checks (EXPLORATORY (post-registration; not part of PREREGISTRATION.md))

Produced by `robustness.py` after adversarial review; none of this is registered. Spearman on the 198 Ma mixtures,
95% percentile CI from a molecule-cluster bootstrap (10,000 resamples, registered seed).

## a. Fitted simple baselines (same 376 Keller training molecules, equal-weight mixture rule)
Features: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, n_S_atoms.

| model | mixture rho [CI] | ensemble minus model [CI] |
|---|---|---|
| registered ensemble | +0.49 [+0.27, +0.65] | |
| ridge (alpha 1, standardised) | +0.39 [+0.10, +0.60] | +0.10 [-0.03, +0.25] |
| random forest (500 trees) | +0.38 [+0.07, +0.62] | +0.11 [-0.05, +0.28] |

## b. Near-duplicate sensitivity (ensemble H2 on mixtures with both components below the Tanimoto cutoff)
| max ECFP4 Tanimoto to training | n mixtures | rho [CI] |
|---|---|---|
| < 0.9 | 166 | +0.51 [+0.27, +0.67] |
| < 0.7 | 113 | +0.53 [+0.28, +0.69] |

## c. Prospective model (Ma and pool molecules removed, 358 training molecules)
On the Ma mixtures: rho +0.46 [+0.22, +0.62]; Spearman with the registered model's mixture predictions 0.959.

## d. P1 with molecule-clustered inference (simulation on the frozen 40 pairs, 48 molecules)
Rule: permutation p < 0.05 AND one-sided 95% cluster-bootstrap lower bound > 0. 2000 simulations, 1000 resamples each.

| true rho (30-rater panel) | permutation alone | rule (power; type-I at 0) |
|---|---|---|
| 0.49 | 0.895 | 0.742 |
| 0.3 | 0.548 | 0.311 |
| 0.0 | 0.077 | 0.019 |

Caveat: Gaussian copula, molecule-level errors independent across molecules; real errors will be messier.
