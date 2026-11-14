# Methods to integrate (only what the opened text supports)

Tags as in VERIFICATION.md. Where a formula was not retrievable it says so; fill from the paper before coding.

## 1. Component pleasantness
**Keller 2017 DREAM** [OPENED-FULL]. Task: predict mean (population) or per-subject pleasantness from Dragon (4884 features) plus Morgan/NSPDK features; scored by Pearson r on 69 hidden molecules, Z-score vs randomised molecule identity. Reported best pleasantness r = 0.41 (individual), 0.71 (population). "Regularized linear models performed nearly as well as random forest" (abstract). Assumes ratings at fixed dilution (1/1000, 1/100,000) on 0-100 scale; pleasantness is only meaningful above the water intensity (14.44, Keller & Vosshall) because unfamiliar/weak stimuli sit near neutral.
Maps to: pleasantness regression baseline and training data (Keller & Vosshall 480 molecules; use 476 to match DREAM).
**Khan 2007** [OPENED-ABSTRACT]: PCA on percepts and on physicochemical descriptors; first percept PC is pleasantness and the first descriptor PC tracks it. Use as a justification for a 1-D pleasantness axis only; no variance numbers verified.
**Arshamian 2022** [OPENED-FULL]: universality ceiling. Variance split: molecule 41%, culture 6%, individual 54% (abstract). Out-of-sample test: random forest from DREAM trained on 466 NYC odorants predicted rank order of 10 held-out odorants in 9 cultures at least as well as a random same-culture participant (Kendall tau). Maps to: noise ceiling for a population-mean predictor and to panel design (rank-based Kendall tau agreement with a held-out panel).
**Lee 2023 POM / OpenPOM** [OPENED-FULL]: predicts 138 odor-quality labels, not pleasantness (the word does not occur in the text). Use as a molecule embedding (256-d penultimate layer) fed to a pleasantness head, not as a pleasantness predictor. POM AUROC 0.89 CV on ~5,000 GS/LF molecules; panel test-retest R = 0.80 (n=15).

## 2. Odorousness gate
**Mayhew 2022** [OPENED-FULL]. Exact rules (Eq 1 and 2): for room-temperature liquids/solids "LogP > -1.72 log VP - 9.10" is the odorous side of the 50% boundary; for gases "LogP < -1.61 log VP + 8.17" (VP in mmHg). Classifier: three transport features (boiling point, vapor pressure, logP) in logistic regression or XGB; Dragon v6 + EPI Suite features; training set 1,924 molecules (84% odorous). Performance: test AUROC 1.0 (30-molecule set), 0.975+-0.028 over 25 resamples of 30; applying both boundaries to the full dataset AUROC 0.88. Assumptions: odorless labels came from lab testing with a safety/availability-limited set; Dragon features for GDB-17 are not redistributable; no gases in the lab test set. Maps to: hard gate before pleasantness (molecule below threshold gets "odorless", abstain from pleasantness).

## 3. POM stack (OpenPOM) availability [OPENED-FULL on README, API listing]
- Repos: github.com/BioMachineLearning/openpom (177 stars) and mirror github.com/ARY2260/openpom (67), both MIT, last push 2025-10-17.
- pip: `pip install openpom` works (PyPI 1.0.0, MIT) but README requires `pip install --pre deepchem` (nightly), torch 2.4.0 from the CUDA 12.4 wheel index, and `dgl` from a version-matched wheel index; Python 3.10; extras pyrfume, dgllife, scikit-multilearn. A CPU-only install is not documented beyond "edit the device_name parameter". Not test-installed here.
- Weights: `models/ensemble_models/experiments_1..10/checkpoint2.pt` are **Git LFS pointers** (each 25,106,835 bytes for experiments_1; the media endpoint answers HTTP 200). So pretrained 10-model ensemble exists in the repo; I did not download it. LFS bandwidth quota and license of the underlying GoodScents/Leffingwell-derived curated CSV (curated_GS_LF_merged_4983.csv) are NOT VERIFIED; the MIT license covers the code.
- Reported: 5-fold CV with 10-model ensemble per fold, ROC-AUC 0.8872 (README), against Lee et al.'s 0.89.
- Not an official release by the Science authors; Lee's own code is only "lightweight reproduction notebooks" at github.com/osmoai/publications.

## 4. Intensity
**Keller & Vosshall** [OPENED-FULL]: intensity rated 0-100 at two dilutions; DREAM intensity r = 0.56 individual, 0.78 population.
**Berglund & Olsson 1993** [OPENED-ABSTRACT]: binary mixture intensity additivity follows the power-function exponents; for binary mixtures additivity "approximate[s] that of the substance with the lower exponent". Use as the structural form (component psychophysical functions then combined) but the equation itself is NOT VERIFIED.
**Stevens power law**: ψ = k·C^n form is the standard; **no odorant exponent was opened**. Swine-slurry paper reported exponent 0.39 (EuropePMC snippet, not a review; do not cite for generality). Stevens 1957 and Cain 1969 are bibliographically verified only.
**Pellegrino 2025** [OPENED-ABSTRACT, preprint]: deep models for intensity of molecules and mixtures from physical properties; claims to beat odor activity values.
Maps to: per-component intensity at the use concentration, needed by the mixture rule below.

## 5. Mixture rule
**Lapid 2008** [text OPENED-FULL, equations NOT VERIFIED]. Structure as stated in text: P_AB is a linear combination of component pleasantness P_A, P_B with weights W_A, W_B computed from perceived intensities I_A, I_B (eq 1). Variants compared: "sin model" (weights squared-sinusoidal in normalised intensity), arithmetic mean, intensity weights, squared weighted intensities. R^2 across four mixtures: sin 0.87-0.94, squared intensities 0.86-0.95, intensity weights 0.83-0.92, arithmetic mean 0.14-0.77. Assumes: binary mixtures, intensities measured for each component at each concentration (paper: "require solid psychophysical data on the separated components"); only explains intermediate pleasantness (mixture between components). The equation images (chemsebjn026fx4..fx10) were not retrievable; get them from the PDF before coding.
**Ma 2020** [OPENED-ABSTRACT]: 222 binary mixtures, 72 odorants, 30 panelists. "pleasantness of the binary mixtures was driven by the pleasantness and intensity of its components"; partial addition in 6 mixtures where components have similar pleasantness; model uses P_A, P_B and "τ-values" for intensity as perceived in the mixture (IAmix, IBmix in the dataset). R^2>0.80, RMSE<0.67 for contrasted pairs. Equation NOT VERIFIED. Dataset is the right benchmark for this stage (CC BY, xlsx, individual rows).
**Snitz 2013 / Ravia 2020** [OPENED-FULL / ABSTRACT]: for **similarity**, not pleasantness. Mixture = sum of component feature vectors, distance = angle between the two vectors; component-wise mean pairwise distance fails. Ravia: 21 features -> angle in radians; <=0.05 rad indistinguishable. Maps to: the mixture embedding if we choose a vector-sum representation and want a perceptual-distance check (metamer test).
**Weiss 2012** [OPENED-ABSTRACT]: ~30 equal-intensity components that span stimulus space converge toward "olfactory white"; implies pleasantness of large mixtures regresses toward a common value. Boundary condition for any component-wise rule beyond ~10 components.
**Lee 2023 caveat** [OPENED-FULL]: POM has no mixture guarantees (quote above). **AtomFormer 2026** [preprint, abstract]: single-pass mixture model, odor-label task only.
**Grosch 2001 / Dunkel 2014** [OPENED-ABSTRACT]: OAV = concentration / threshold; recombination and omission tests; "exceptions where odorants with high OAVs are suppressed". Use OAV as a screening prior on which components matter, not as an additive rule. Maps to: component selection for food aromas, and to the omission-test validation design.

## 6. Abstention / applicability domain
**Sheridan 2004** [OPENED-ABSTRACT]: error is lower for molecules with high similarity to the nearest training molecule and/or many neighbours above a user-chosen similarity cutoff, independent of QSAR method or descriptor. Implementation: nearest-neighbour Tanimoto (or POM-space distance) to the 476/480 training set; abstain above distance cutoff.
**Tropsha 2010** [OPENED-ABSTRACT]: requires defined applicability domain and external validation. Cutoff value not verified.
**Split conformal** [Angelopoulos & Bates, OPENED-FULL]: calibrate score s_i on n held-out points; q-hat = the ceil((n+1)(1-alpha))/n quantile; C(x) = {y : s(x,y) <= q-hat}; guarantee P(Y_test in C(X_test)) >= 1-alpha under exchangeability (iid in the theorem). Marginal only. **Class-conditional (Mondrian-style)**: stratify calibration scores by class, q-hat(k) = Quantile(s^(k); ceil((n_k+1)(1-alpha))/n_k), with n_k the count in class k; yields P(Y in C | Y=y) >= 1-alpha. The text also has a group-balanced version conditioned on a covariate. Vovk 2013 is the formal source for Mondrian/conditional validity; not opened.
Assumptions to check: exchangeability fails if calibration molecules differ from deployment molecules (food volatiles vs Keller's 480); the covariate-shift variant (Thm 3) exists. For regression of pleasantness the score would be the absolute residual; this regression recipe was not located in the opened text, so treat as NOT VERIFIED for citation.
Maps to: abstention = empty/wide prediction set or applicability-domain failure; stratify by odor class or by Mayhew gate.

## 7. Noise ceiling
**DREAM** [OPENED-FULL]: 20 molecules rated twice; "test-retest correlation across individuals and molecules" per attribute is the upper limit; no model should do better than "data from an independent trial with an identical stimulus and individual". **Lee 2023**: panel test-retest R = 0.80 (n=15) for mean ratings; DREAM cohort noisier. **Arshamian**: only 41% of individual rank variance is molecule-level, so an individual-level r near 0.64 (sqrt 0.41) is the structural ceiling for a molecule-only predictor; this is my arithmetic from their figure, not their statement.

## 8. Panel design
Measured designs to copy: Keller & Vosshall (55 retained subjects, bottom 10% by acuity ranking excluded, 20 repeats); Lapid (84 subjects, 5 ratios, olfactometer 6 L/min, 2 s pulse, 30 s ITI); Ma (30 trained assessors, outlier subject removed); Arshamian (rank 10 odorants, n=16-55 per culture); Lee (trained panel n=15).
**Triangle/Thurstonian**: Ennis 1993 record verified only; d' and power formulas NOT VERIFIED; ISO 4120 edition NOT VERIFIED. Bushdid's three-vial forced choice is the same paradigm, but its resolution inference is invalid (see critiques).

## 9. Food-pairing / compound-sharing prior
Ahn 2011 [OPENED-FULL]: bipartite ingredient-compound network (381 x 1,021), compound sharing score; Western positive, East Asian negative. Jain 2015 [OPENED-FULL]: Indian cuisines negative. Spence 2020 critique content NOT VERIFIED. Conclusion we can support: compound sharing does not predict acceptability universally; do not use it as a pleasantness prior.

## 10. Receptor-level features (optional)
Mainland 2015 [OPENED-ABSTRACT]: 73 odorants x 511 ORs luciferase screen. Lalis M2OR [OPENED-FULL]: 75,050 assays/51,395 pairs, with mixture-class entries (space-separated InChIKeys). Too sparse for pleasantness regression; relevant only to an OR-activation embedding.

## 11. Critiques that bound our claims
- Bushdid 2014: Gerkin & Castro [OPENED-FULL abstract]: estimate is an upper bound, "no evidence for the original claim". Meister: "This claim is wrong." Science 2024 formal Erratum exists. Do not use "trillion discriminable stimuli" or discriminability resolution derived from it.
- Lee 2023: label prediction, not pleasantness or mixtures; authors state no mixture guarantees.
- Snitz/Ravia: similarity of mixtures, evaluated on mixtures of 4-43 components of intensity-matched odorants.
