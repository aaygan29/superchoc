# Results: structure-only prediction of Ma 2021 mixture pleasantness

Pre-registered plan: `PREREGISTRATION.md` (61a9465). Code: `mixture_hedonics.py` (sha256 in JSON; run on HEAD 7886a77 with the
file uncommitted). All numbers: `results/mixture_hedonics.json`; per-mixture: `results/mixture_hedonics_predictions.csv`.
Run: 10,000 molecule-cluster bootstrap resamples (seed 20261113), 1,000 label permutations, model seed 0, 993 s.

Data: Keller 1/1000 = 420 molecules; 44 removed as Ma molecules (42 by CID, 44 by stereo-stripped SMILES; union 44),
376 trained on, zero overlap asserted. Ma: 72/72 molecules valid, 198/198 mixtures used, no exclusions.
"Liked" = above median 4.97 (99 of 198).

## Hypotheses (Holm across H1-H3, alpha 0.05)

| | Spearman / AUROC (95% CI) | raw p | Holm p | Verdict |
|---|---|---|---|---|
| H1 components vs measured alone rating (n=72) | 0.640 (0.472, 0.763) | 0.001 | 0.003 | **Supported** |
| H2 mixtures (n=198), permutation | 0.490 (0.269, 0.647) | 0.001 | 0.003 | **Supported** |
| H3 AUROC confident (n=83) | 0.686 (0.521, 0.867) | 0.417 | 0.417 | **Not supported** (see below) |
| H3 AUROC abstained (n=115) | 0.667 (0.487, 0.824) | | | |

H3: the literal criteria (confident AUROC > abstained AUROC; confident CI lower bound > 0.5) are both met, but the gap is
0.019 (CI -0.20, 0.26): abstention did not isolate better predictions. With Holm applied (p defined in Deviation 2) H3 is not
significant. Treat H3 as not supported; the abstention signal carries no demonstrated value. AUROC over all mixtures
is 0.710 (0.601, 0.811).
The permutation null centres near 0, so H1/H2 reflect real signal from structure; p floor is 1/1001.

## Secondary analyses (descriptive)

| | Mixture Spearman (95% CI) | Component Spearman |
|---|---|---|
| A1 RF ECFP4 | 0.350 (0.190, 0.500) | 0.566 |
| A1 RF descriptors | 0.429 (0.148, 0.637) | 0.593 |
| A1 HGB descriptors | 0.483 (0.209, 0.679) | 0.635 |
| A1 kNN Tanimoto | 0.451 (0.267, 0.594) | 0.598 |
| A1 Ensemble | 0.490 (0.269, 0.647) | 0.640 |
| A2 molecular weight | 0.126 (-0.075, 0.343) | 0.280 |
| A2 heavy atoms | 0.162 (-0.048, 0.382) | 0.345 |
| A2 constant | undefined (AUROC 0.5) | |
| A3 oracle, measured component ratings | 0.897 (0.828, 0.938); Pearson 0.914 | |
| A4 intensity-weighted, predicted components | 0.518 (0.284, 0.680) | |

- A2: ensemble beats MW by 0.364 (0.147, 0.544) and heavy atoms by 0.328 (0.137, 0.485); no baseline matches it. Single
  families are not distinguishable from the ensemble (HGB 0.483 is on par); the ensemble gives no demonstrated gain.
- A3: oracle is higher by 0.407 (0.247, 0.616): most of the gap to the ceiling is component prediction, not the mixture rule.
- A4: weighting by measured intensity-in-mixture adds 0.029 (diff CI -0.054, 0.007 for ensemble minus A4): no clear gain.
- A5: in-domain (both components max Tanimoto >= 0.4; n=151) 0.492 (0.204, 0.681); out-of-domain (n=47) 0.411
  (-0.136, 0.788; 48 resamples undefined). No evidence of a difference; out-of-domain is too imprecise to say.
- A6: Keller 0-100 (mean 45.2, SD 12.7) vs Ma 0-10 (mean 4.91, SD 0.99, narrow range); only rank statistics used.

## Deviations / interpretations
1. Holm p-values: H1 and H2 both from the label-permutation test (plan names it only for H2); H1 gets no bootstrap p.
2. H3 p-value: bootstrap one-sided, max of P(AUROC_conf <= 0.5) and P(AUROC_conf - AUROC_abst <= 0). Verdict requires criterion AND Holm p < 0.05.
3. Percentile CIs. Mixtures kept once if both components drawn (not multiplicity-weighted); H1 keeps drawn molecules with multiplicity.
4. Rank normalisation = mid-rank ECDF of each family's in-sample Keller-training predictions.
5. "Keller median" = median of Keller training labels (45.8); families' raw predictions compared with it; ties count as below. Confident = all four agree for A and for B (A and B may be on opposite sides).
6. "Liked" strictly above the median. Duplicate pairs: the supplied 198-row panel-mean table is used; H1 molecule value = mean of per-mixture alone ratings across all mixtures with it.
7. CAS-named component (CID 61592, ethyl 3-(methylthio)propionate) resolved by CID; matching by CID throughout.
8. Descriptors: inf -> NaN, NaN -> training median; HGB random_state=0.
9. A4 weights = panel-mean IAmix/IBmix from behavior.csv (zeros kept, linear weights). A2 baselines unsigned. A3 uses per-row PA/PB.
10. A2 "matches within CI" operationalised as a paired bootstrap difference CI; A5 "in domain" = both components >= 0.4.
11. Permutations refit all four families (and rank normalisation) on shuffled labels, seeds 1,000,000+i.

## Limits
- Ma's components are near-isointense food odorants with a narrow pleasantness range (SD ~0.9); no strongly aversive components.
- 72 molecules: CIs are wide (H2 lower bound 0.27) and mixtures share components.
- Predicts the panel mean, not individuals. Replace-the-code risk: results are for the sha256 recorded; code not committed.
- Reuse: `build_component_ensemble(exclude_cids)` and `predict_components(smiles_list)` (default ensemble = Ma-excluded, identical to the one validated).
