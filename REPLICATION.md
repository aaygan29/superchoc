# Replication guide

Every step needed to reproduce the results in this repository, and the methodology behind
them, so anyone can rerun, check, or extend the work. Conjecture and framing are Jake
Wintermute's (see [README](README.md)); the methods and analysis here are the author's.

## Environment

- Python 3.9+.
- Core: `numpy`, `scikit-learn`, `scipy`. Real featurization: `rdkit`. Live ingestion:
  standard library only (`urllib`).
- No GPU required. All results below run on a laptop CPU in minutes.

```bash
pip install numpy scikit-learn scipy rdkit
```

Everything is seeded; results are deterministic given the seeds stated.

## What is validated, and against what

Nothing here is validated on real flavor. Two methods are validated on a **reproducible
synthetic ground truth** (`FlavorSpace` / `ground_truth.py`): a POM-style embedding mapped
to a noisy "deliciousness" surface. This is a fair test of the *methods* (do they beat the
obvious baseline?), not a claim about real taste. Real claims require the public datasets
in [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md) and, for deliciousness, blinded human panels.

## Method 1: online active flavor search

`methods/active_flavor_search/`. Question: does sequential (online) experimental design
pick better molecules to taste than random screening?

```bash
cd methods/active_flavor_search
python test_active_search.py     # 6 unit tests
python validate.py               # prints metrics + writes results/validation.json
```

Method: lift the flavor embedding into Random Fourier Features (approximating an RBF-kernel
Gaussian process), keep a closed-form Bayesian linear model updated after every tasting,
and pick the next molecule by Upper Confidence Bound. Result (25 seeds, budget 50):
UCB beats random by +0.074 mean best-found, wins 25/25 seeds, and at half budget already
beats random at full budget. Thompson sampling is included but **not validated** here
(over-explores; see that folder's README).

## Method 2: goodness model + combo generator

`methods/goodness_model/`. Question: can we learn which molecules are "good" and compose
novel combinations that are genuinely good, not just good in the model's opinion?

```bash
cd methods/goodness_model
python test_goodness.py          # 4 unit tests (incl. real RDKit featurizer)
python validate_goodness.py      # prints metrics + writes results/validation_goodness.json
python generate_suite.py > suite.json   # ranked candidate suite (also results/candidate_suite.md)
```

Pipeline and result are documented in that folder's README. The load-bearing check is the
**Goodhart control**: combos chosen to maximize the model's score are also good on the true
oracle (population ranking corr 0.86; the model under-promises its picks by 0.05).

### Exact methodology (Method 2, step by step)

1. **Ground truth.** `FlavorSpace(seed)` builds a fixed pool of molecule embeddings and a
   mixture-of-Gaussians deliciousness surface. `query()` returns a noisy tasting;
   `true_pool` / `true_mixture()` are the noiseless oracle used only for scoring.
2. **Labels.** Sample 400 molecules, observe noisy goodness (`build_training_set`).
3. **Model.** Fit a `RandomForestRegressor` (200 trees) on embedding -> goodness; score
   held-out R2 against the noiseless surface (`evaluate_model`).
4. **Chocolate reference.** The 88th-percentile random k-combo (excellent, not maximal),
   per the conjecture's claim that chocolate is not the peak.
5. **Generation.** From random k-molecule combos, hill-climb by single-molecule swaps on
   predicted goodness, keeping only combos whose mixture embedding is >= 0.2 cosine
   distance from the chocolate reference (novelty). `compose_combos`.
6. **Ranking + suite.** Rank candidates by predicted goodness; emit JSON + markdown with
   predicted goodness, true-oracle goodness, novelty, and "beats chocolate" flag.
7. **Validation.** Pre-registered checks (learnable model, beat random, beat chocolate,
   two Goodhart controls, novelty) over 8 seeds. Ships only if all pass.

## From synthetic to real (how to make a lab-testable suite)

1. **Get data.** Cache the public sources with the loaders in `data/` (Pyrfume,
   GoodScents/Leffingwell, DREAM, ChEMBL, GPCRdb) or pull live with
   `data/online_ingest.py` (ChEMBL, PubMed, bioRxiv REST APIs; no key needed).
2. **Featurize.** Replace synthetic embeddings with ECFP4 fingerprints
   (`featurize.smiles_to_ecfp4`) or POM/GNN embeddings.
3. **Retrain goodness** on real odor/pleasantness labels; keep the same held-out protocol.
4. **Replace the mixture aggregator** (mean-pooling) with a learned mixture model, because
   real mixtures are non-additive (Duchamp-Viret et al. 2003).
5. **Regenerate the suite** with a molecule table so members are named compounds with
   SMILES and sourcing notes.
6. **Test for real.** Synthesize/blend the top candidates and run a blinded taste panel.
   Only the panel closes the "as delicious as chocolate" claim.

## Provenance of the results in this repo

- `methods/active_flavor_search/results/validation.json` - Method 1, 25 seeds, budget 50.
- `methods/goodness_model/results/validation_goodness.json` - Method 2, 8 seeds, k=4.
- `methods/goodness_model/results/candidate_suite.{json,md}` - example suite, seed 0
  (synthetic demo; banner in-file).

Regenerate any of them by running the commands above; they are deterministic.
