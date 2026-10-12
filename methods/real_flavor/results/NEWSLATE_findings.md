# NEWSLATE findings (branch newslate-evidence). Every number names its output file in results/newslate/.

Status log: partial results written as computed. Sections marked PENDING are not yet run.

## (a) J specification (configs/J_spec.json, newslate/J_SPEC.md)
- Legacy code (denovo_recipes.compose) implements J = 1.0*T + 0.15*C + 0.4*N, T = mean predicted pleasantness/100, C = mean composition score,
  N = chemical novelty. Stochastic first-improvement 1-swap hill climbing, 5 restarts x 500 proposals, no stopping rule beyond the budget.
  The paper text defined none of this. Legacy taste is IN-SAMPLE for known molecules (full-data RF predicts its own training molecules).
- newslate/jspec.py reimplements it exactly: with in-sample taste and w_mix=0 it reproduces all 5 legacy recipe sets and J values
  (test_jspec.json, legacy_compose.json; max |N_fast - N_legacy| = 5e-9).

## (e) Splits (audit_claims.json)
- Repo CV (goodness_real.cross_validate) is RANDOM KFold over molecules. Random 5-fold Spearman 0.486 (mean of 3 seeds; 0.50 in validation_real.json), R2 0.263.
- Butina cluster split (sim>=0.4, 146 clusters): Spearman 0.406, R2 0.224. Generic-Murcko scaffold split (36 scaffolds, 47% acyclic share the empty scaffold): Spearman 0.302, R2 0.122.
- Mean max-Tanimoto test-to-train: 0.584 random, 0.472 Butina, 0.359 scaffold. Random-CV Spearman by max-train-similarity bin: 0.32 (<0.3), 0.42, 0.55, 0.42, 0.51 (>=0.7), so within random CV the dependence on similarity is weak, but the split-level drop is large.
- De novo candidates sit at mean max-Tanimoto to Keller 0.41 (generator_baselines.json), i.e. in the regime where the scaffold split (0.30) is the relevant number.

## (f) Receptor claim (audit_claims.json)
- 1.57x = 9.1% nearest-neighbour shared-receptor rate vs 5.8% chance on n=22 ligand-receptor pairs (2 hits of 22; 95% bootstrap CI of the rate 0 to 22.7%).
  The 2 hits are ONE mutual pair (creosol and vanillin, both on OR10G4, Tanimoto 0.53). Binomial test of 2/22 vs 5.8%: p=0.37. Against a self-pair-excluded
  chance of 1.3% the lift is 7.0x and p=0.033, but that is one ligand pair. Not evidence of a usable receptor layer.
- receptor_in_J_check.json: M2OR overlaps Keller on only 38 molecules; Spearman(#responsive receptors, pleasantness) = 0.14 (p=0.42). The receptor layer cannot enter J as a
  validated term (no measured receptor-to-percept map). Decision: DROP the receptor contribution claim.

## (h) Lean (audit_claims.json)
- `lean FlavorMath.lean` exits 0. The theorems state that each element of a List Nat is <= listMax (iterated max). That is the definition of max.
  No weights, convexity, or real-valued T(R) are formalized. Footnote, not a results-table row.

## (i) Concentration (audit_claims.json; scope limit)
- Nothing in J or the models takes concentration. In the repo's own Keller 2016 data, 1/1000 vs 1/10 pleasantness: Spearman 0.63 (n=134), mean absolute change 8.2 points,
  29% of molecules change by >=10 points; 1/1000 vs 1/100000: Spearman 0.68 (n=288), 8.1 points, 30% >=10 points. Addressing it needs dose-response labels per component
  (OAV/threshold data) and a concentration-aware mixture model; Keller has 2 usable dilutions per molecule, which is not enough to fit one.

## (d) Mixture model on REAL held-out human data (real_mixture_eval.json, mixture_disjoint_transfer.json, mixture_leakage_check.json)
- Synthetic oracle reproduced: mean-pool 0.249, set 0.819 (5 oracle seeds). SAME code with an ADDITIVE oracle (suppression 0): mean-pool 0.840, set 0.817. The gap is built in by construction.
- Real data (Spearman vs human similarity; set-RF minus meanpool-RF, 95% pair-bootstrap CI):
  train Snitz, test Ravia (n=195): meanpool 0.291, set 0.323, diff +0.032 [-0.061, 0.124]; untrained angle 0.190.
  train Ravia, test Snitz (n=360): meanpool 0.502, set 0.476, diff -0.026 [-0.103, 0.053]; untrained angle 0.492.
  train Ravia (molecule-disjoint from Snitz, 100 pairs), test Snitz: meanpool 0.444, set 0.442, diff -0.001 [-0.074, 0.073].
  Snitz leave-mixture-out: 0.583 vs 0.621, diff +0.038 [-0.007, 0.087]. Ravia leave-mixture-out: 0.125 vs 0.413, diff +0.288 [0.153, 0.422], BUT 97% of held-out pair molecules
  also occur in training pairs (Ravia has only 28 mixtures), so that figure is leakage-prone and not counted as support.
- Controls: label-permuted set model gives 0.04 / -0.33 / 0.06 / -0.20 (chance or below); a size-only model (component count) reaches 0.31 / 0.41 / 0.54 / 0.23, so much learnable signal is mixture size.
- Verdict: the advantage of non-additive pooling VANISHES on clean held-out real data (differences within CI of zero; learned models barely beat the untrained additive angle in transfer).
  Neither the 0.82 nor the claim that the learned mixture model beats additive pooling is supported. Refuted on real data.

## (g) Synthesizability and generator baselines (generator_baselines.json)
- SA score (RDKit/Ertl): all 434 candidates mean 2.21 (0% > 6), top-40 guided 2.38, known Keller molecules 2.13. SA does not separate anything: BRICS recombination of small
  flavor molecules is trivially "easy" by this score, and SA says nothing about stability or whether a route exists. All 12 PubChem-checked candidates were absent (database novelty).
- Guided top-40 (ranked by predicted pleasantness) vs unguided draws from the same enumeration (x300): predicted pleasantness +12.4 under the ranking model, but only +1.2 under an independent
  descriptor GBM (unguided draw sd 0.73). Correlation of the two predictors on the 434 candidates is 0.27. Most of the guided gain is selection on the predictor itself.
- Re-seeded unguided BRICS (8 seeds): predicted pleasantness 43.1 vs 52.5 for the pipeline's enumeration, joint-pass 0.14 vs 0.25 (unguided same enumeration) vs 0.975 guided (selected on the criterion, circular).

## (b) Wiring M into J (generation_analysis.json, 110 runs over results/newslate/generation_raw.json)
- M is now in J (jspec.py, J_spec.json: J = w_t*T + w_c*C + w_n*N + w_mix*M, w_mix=0.4 fixed a priori, not tuned).
  A code bug that had stopped this step from ever running was fixed (analyze_generation.py imported rdFingerprintGenerator from `rdkit` instead of `rdkit.Chem`).
- Adding the mixture term barely changes what is generated. Full-J vs additive-J recipe sets (same seed): top-recipe Jaccard 0.53, union Jaccard 0.59,
  against a seed-to-seed noise floor of ~0.49 to 0.51. So turning M on moves the output only marginally above the noise from changing the random seed.
- M in practice is nearly additive anyway: on generated recipes mean M minus its additive aggregate is 0.005 (full-J) / -0.007 (additive-J), |gap| ~0.011,
  and on 300 random feasible blends M-minus-additive is 0.003 (|gap| 0.027). The learned "non-additive" term rarely departs from an additive pooling of the same features.

## (c) Weight ablation (2^4 factorial, 5 seeds per cell; generation_analysis.json)
- Factorial main effect of each term on the predicted-taste metric T: taste +0.137 (standardized 1.56), composition +0.072 (0.82), novelty -0.035 (-0.40),
  mixture +0.0002 (0.002, CI through zero). On every output metric the mixture main effect is within the 95% CI of zero.
- Drop-one influence (Jaccard of the drop-one recipe set vs the full set, same seed; LOWER = more influential): taste 0.26, composition 0.36, novelty 0.52, mixture 0.61,
  vs noise floor 0.49. Ranking of influence: taste > composition > novelty > mixture, with mixture essentially at the noise floor.
- Verdict: taste and composition drive the generated recipes; novelty reshapes them modestly; the mixture term is the least influential and does not meaningfully change outputs.
  Consistent with (d): the non-additive mixture signal does not do real work on this dataset. Figure: figures/fig1_weight_ablation.pdf. infeasible_init total = 0.
