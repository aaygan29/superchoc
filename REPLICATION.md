# Replication guide

Everything runs on a laptop CPU. Seeds are fixed; outputs are deterministic.

## Environment

```bash
uv venv && uv pip install -r requirements.txt   # numpy, scipy, scikit-learn, rdkit, pyrfume, joblib, pytest
export OMP_NUM_THREADS=1 PYTHONPATH=methods
python -m pytest -q methods
```

Data is fetched from the pyrfume archive on first use (`keller_2016`, `ma_2021`,
`bushdid_2014`, `snitz_2013`, `ravia_2020`, `leffingwell`) and cached locally.

## 1. The pre-registered analysis

Plan: `methods/chocolate_anchor/PREREGISTRATION.md`, commit `61a9465` (pushed to GitHub
before the analysis ran). To check the plan was not changed afterwards:

```bash
git log --follow --format='%h %ci' -- methods/chocolate_anchor/PREREGISTRATION.md
```

```bash
python -m chocolate_anchor.mixture_hedonics     # ~16 min: 10,000 bootstraps, 1,000 permutations
```

Writes `results/mixture_hedonics.json` and `results/mixture_hedonics_predictions.csv`. The
summary, including every interpretation of the plan, is in `RESULTS_MIXTURE_HEDONICS.md`.
Key controls: test molecules removed from training by CID and stereo-stripped SMILES, with
an assertion; confidence intervals resample molecules, not mixtures, because mixtures share
components; the permutation test retrains on shuffled training labels.

## 2. Exploratory analyses (designed after registration)

```bash
python -m chocolate_anchor.exploratory           # noise ceiling, conformal abstention, power
python -m chocolate_anchor.discrimination        # triangle-test model on Bushdid 2014
```

## 3. Prospective predictions

```bash
python -m chocolate_anchor.candidate_pool        # offline from data/candidates/evidence.json; --refresh re-fetches
python -m chocolate_anchor.prospective           # scores all pairs, writes FROZEN_PREDICTIONS.csv
shasum -a 256 data/candidates/FROZEN_PREDICTIONS.csv   # must match FROZEN_PREDICTIONS.sha256
```

## Version 1

The original pipeline and its results run at tag `v1-archive`; what was removed and why is
in `docs/REMOVED.md`.
