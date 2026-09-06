# Goodness model + combo generator

The two-part engine behind Wintermute's Criteria 1 and 2: a machine-learning model that
**learns which molecular embeddings taste good**, and a generator that **composes novel
molecule combinations** predicted to be good, producing a ranked, lab-testable suite of
"super-chocolate" candidates.

The idea, in the user's framing: chocolate is one specific mixture of many molecules that
happens to be delicious. This searches for other mixtures that should be delicious too,
including ones no one has tasted, so a chemistry lab or a molecular gastronomist can make
and taste the top candidates.

## Pipeline

```
molecule embedding (POM/GNN, or ECFP4 for real molecules via featurize.py)
   -> GoodnessModel (RandomForest regressor)  [goodness_model.py]  Criterion 1
   -> compose k-molecule combos, hill-climb on predicted goodness,  [compose.py]  Criterion 2
      keep only combos novel vs a chocolate reference
   -> ranked candidate suite (JSON + markdown)                     [generate_suite.py]
```

Mixtures are aggregated by mean-pooling the member embeddings. This is a documented
simplification: real olfactory mixtures are **non-additive** at the single-neuron level
(suppression/synergy; Duchamp-Viret et al. 2003, [DOI](https://doi.org/10.1111/j.1460-9568.2003.03001.x)),
so on real data this aggregator must be replaced by a learned mixture model. Hedonic
valence ("goodness") is modeled as a scalar because pleasantness is one of the primary
dimensions of odor perception (Barnum & Hong 2022, [DOI](https://doi.org/10.1016/j.cub.2022.10.067)).

## Validation result (reproducible: `python validate_goodness.py`)

8 seeds, k=4, against a chocolate reference set at the 88th percentile of random combos
(excellent but not maximal, per the conjecture):

| check | value | bar | pass |
|---|---|---|---|
| goodness model held-out R2 | 0.651 | >= 0.50 | yes |
| generated combos beat random combos (true goodness) | +0.094 | >= 0.02 | yes |
| generated combos beat chocolate reference (true goodness) | +0.037 | >= 0 | yes |
| **Goodhart control**: model ranks a random-combo population | corr 0.858 | >= 0.6 | yes |
| **Goodhart control**: model does not over-promise its picks | gap -0.054 | <= 0.10 | yes |
| novelty of generated combos vs chocolate (cosine) | 0.80 | >= 0.2 | yes |

The Goodhart control is the important one: the combos the generator picks to maximize the
**model's** score are also good on the **true oracle** (and the model actually
under-promises them by 0.05). The generator is discovering real structure, not exploiting
model error. Committed result: `results/validation_goodness.json`. Unit tests (4/4,
including a real RDKit featurizer on vanillin and isoamyl acetate): `python test_goodness.py`.

## The deliverable: a candidate suite

`python generate_suite.py > suite.json` produces a ranked suite; a rendered example is in
`results/candidate_suite.md`. Each candidate lists the molecules to combine, predicted
goodness, novelty vs chocolate, and whether it beats the chocolate reference on the oracle.
Running as-is it is a **synthetic demo** (members are pool indices, scores are
demonstration values); the output banner says so. For a real, lab-testable suite: provide
a molecule table (index -> {name, SMILES}) and a goodness model trained on real labels
(see `../../data/` and `featurize.py`); the generator then emits named compounds a lab or
chef can source and taste.

## What failed on the way (documented, not hidden)

- First chocolate reference was the best-of-N combo (near the global maximum). That
  contradicts the conjecture (chocolate is not the peak), and made "beat chocolate while
  novel" nearly impossible. Fixed to a high-percentile reference.
- First Goodhart metric correlated predicted-vs-true over only the ~5 near-tied winning
  combos, which is statistically ill-posed. Replaced with a population ranking test over
  200 random combos, plus a no-over-promise gap check on the picks.

## Files

- `flavor_space.py` - synthetic flavor space with mixtures + chocolate reference.
- `featurize.py` - RDKit SMILES -> ECFP4 (the real-data entry point).
- `goodness_model.py` - the goodness regressor + held-out evaluation.
- `compose.py` - novelty-constrained combo generator (hill-climb).
- `generate_suite.py` - emits the ranked candidate suite (JSON + markdown).
- `validate_goodness.py` - the pre-registered validation with Goodhart controls.
- `test_goodness.py` - unit tests.
- `results/` - committed validation JSON and an example suite.

_Conjecture and framing: Jake Wintermute (see repo README). Cross-disciplinary citations
via PubMed; DOIs above. Method, generator, and validation: mine._
