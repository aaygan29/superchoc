# EXPLORATORY analyses (post-registration)

**Not pre-registered.** Run after the registered results (`RESULTS_MIXTURE_HEDONICS.md`), which are untouched and read-only here. Code `exploratory.py`, numbers `results/exploratory.json`. Citation keys are in `docs/literature/REFERENCES.bib`.

## 1. Noise ceiling (Keller2017dream-style test-retest logic; Spearman-Brown)
Method: 1000 random halves of Ma's 60 raters, half-panel means, Spearman over targets, Spearman-Brown to the full panel (R); ceiling = sqrt(R) (max rho of a noise-free predictor). Cites `keller2017dream` (noise ceiling as upper limit), `ma2021data` (per-rater data).
- Mixtures: R = 0.88 (split CI 0.85-0.91); ceiling rho = 0.94 (split CI 0.92-0.95).
  H2 rho 0.490 = 52% of ceiling; oracle A3 0.897 = 96%.
- Components: R = 0.92 (split CI 0.88-0.95); ceiling rho = 0.96 (split CI 0.94-0.98); H1 rho 0.640 = 67% of ceiling.
- CIs are split-to-split spread only (excludes panel sampling uncertainty). Raters rate different subsets (~30 of 60 per mixture): halves are formed over raters globally; each target uses whichever raters of the half rated it.

## 2. Split-conformal abstention (Angelopoulos & Bates, `angelopoulos2023`)
Quantile q-hat = ceil((n+1)(1-a))/n quantile (verified, opened). Score = |obs rank - pred rank| on the calibration-ECDF rank scale; the absolute-residual regression recipe itself was NOT located in the opened text (VERIFICATION.md), so it is the standard choice, not a verified citation. 200 molecule-disjoint splits (72 molecules halved; calibration mixtures have both components in half 1, test both in half 2; n_cal ~48, n_test ~49, ~101 straddling dropped). Decision: liked = interval wholly above the median threshold, disliked = wholly below, else abstain. Mondrian = class-conditional q-hat per observed class (`angelopoulos2023` class-conditional recipe, opened; `vovk2013conditional` not opened), label kept if its interval reaches its region.

| a | cover (mean, 5-95%) | width | 3-way decided / acc | Mondrian decided / acc (cover liked/disl.) | H3 rule decided / acc | no abstention acc |
|---|---|---|---|---|---|---|
| 0.2 (target 0.80) | 0.79 (0.60-0.94) | 0.62 | 21% / 0.73 | 18% / 0.73 (0.80/0.80) | 40% / 0.66 | 0.63 |
| 0.1 (target 0.90) | 0.89 (0.76-1.00) | 0.74 | 6% / 0.77 | 4% / 0.74 (0.91/0.90) | 40% / 0.66 | 0.63 |
Coverage is marginal over test mixtures that share components; exchangeability is only approximate.

## 3. Odorousness gate (`mayhew2022`)
Gate implemented (mayhew2022 boundary equations, verified) but not validated or applied: it needs a sourced vapor pressure per molecule, which this repo does not have. Not needed for the prospective study: every candidate is a listed flavoring, i.e. odorous in use.

## 4. Power for a prospective Ma-style panel (simulation; assumptions in JSON)
Resamples Ma's data structure (sigma_mu 0.95, sigma_e 1.64; Ma's empirical rater residuals), Gaussian-copula predictions, alpha 0.05 one-sided, 2000 sims/cell. Targets: registered H2 rho 0.49 and 0.30 as the expected Spearman at a 30-rater panel (disattenuated by Ma's split-half R). Cells: Spearman power by n_mixtures (rows) x n_raters 20/30/40; classifier-accuracy power (>0.5 on decided) in brackets at decided share 0.21 (conformal 3-way) | 0.42 (H3 confident rate).
- target_rho=0.490, latent rho 0.52:
  n=20: 20: 0.70 [0.00|0.42]; 30: 0.74 [0.00|0.47]; 40: 0.74 [0.00|0.45]
  n=30: 20: 0.88 [0.31|0.66]; 30: 0.90 [0.34|0.66]; 40: 0.89 [0.34|0.66]
  n=40: 20: 0.95 [0.58|0.65]; 30: 0.96 [0.60|0.68]; 40: 0.96 [0.61|0.68]
  n=60: 20: 0.99 [0.83|0.81]; 30: 1.00 [0.84|0.83]; 40: 1.00 [0.85|0.85]
- target_rho=0.300, latent rho 0.32:
  n=20: 20: 0.34 [0.00|0.20]; 30: 0.36 [0.00|0.22]; 40: 0.35 [0.00|0.21]
  n=30: 20: 0.47 [0.12|0.32]; 30: 0.48 [0.13|0.32]; 40: 0.49 [0.14|0.34]
  n=40: 20: 0.59 [0.27|0.29]; 30: 0.62 [0.29|0.28]; 40: 0.61 [0.29|0.30]
  n=60: 20: 0.75 [0.45|0.39]; 30: 0.76 [0.43|0.36]; 40: 0.79 [0.46|0.41]
The design_matched disattenuation variant is in the JSON.
