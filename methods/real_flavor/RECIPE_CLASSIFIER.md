# Recipe-level "good flavor" classifier: `recipe_classifier.py`

The per-molecule goodness model scores ONE molecule. This module learns at the RECIPE level:
tell a real good flavor combo (chocolate, vanilla, coffee, banana, strawberry, rose, plus
single high-pleasantness molecules like vanillin) apart from an arbitrary random blend. That
gives a single non-additive "good flavor" response over a mixture, which is what makes
optimal-ratio search meaningful (see below).

## Data
- **Positives:** the six reference flavors and random subsets of each (a subset of a good
  flavor's key odorants is still flavor-like), plus top-quartile-pleasantness single Keller
  molecules.
- **Negatives:** size-matched random blends of Keller molecules, plus bottom-quartile singles.

## Features (mixture-level, proportion-aware)
Snitz physicochemical mixture vector (`mixtures.MixtureSpace`) + composition class fractions +
component count + olfactory-white risk + goodness-model pleasantness (mean/max/min of parts).
Because the mixture vector depends on the component proportions, the classifier score is a
genuine function of the ratios.

## Models
RandomForest + GradientBoosting + LogisticRegression, plus a soft-vote ensemble.

## Validation: leave-one-flavor-out ROC-AUC (`results/recipe_classifier.json`)

Train on five flavors (+ singles + randoms), test on the held-out flavor's positives vs random
negatives, so a flavor's subsets cannot leak between train and test. This tests generalization
to an UNSEEN flavor class.

| model | LOFO ROC-AUC |
|---|---|
| random forest | 0.67 |
| gradient boosting | 0.49 (at chance) |
| logistic | 0.61 |
| **ensemble** | **0.69** |

Honest reading: generalization to a brand-new flavor class is **modest** (ensemble 0.69), and
gradient boosting is at chance. In-distribution the deployed classifier behaves sensibly (vanilla
0.98, chocolate 0.88, a vanillin+menthol combo 0.65, a random blend 0.02), but that is easier
than the held-out test. **Most important caveat:** positives are "known real flavors /
high-pleasantness", negatives are "random blends", so the classifier learns flavor-like
COHERENCE, not proven deliciousness. There is no human combo-deliciousness ground truth; a high
score means "looks like a real flavor", a useful ranking prior, not a guarantee it tastes good.

## Why this matters: it fixes ratio optimization

An additive response optimizes to a single component (the `FlavorMath.lean` best-part bound), so
Scheffe ratio optimization collapsed to one molecule. The classifier score is proportion-aware
and non-additive, so optimizing it over the simplex yields real interior blends. Example
(chocolate odorants): ratios `[0.49, 0.01, 0.17, 0.01, 0.32]`, interior optimum, design R^2 0.82.
For a recipe the classifier does not recognize as flavor-like, the optimum can still collapse to
a vertex, and `optimal_levels.interior_optimum` reports that honestly.

## Wired into the pipeline (the "full package")

`denovo_recipes.compose()` fits the classifier once and attaches to each recipe:
`good_flavor_probability`, and for the top recipe `optimal_levels` derived by maximizing the
classifier's P(good flavor) over the simplex. Combined with the existing fields, each recipe is
now a full package: components (name, SMILES, source, safety, odor notes), predicted
pleasantness, chemical + perceptual novelty, flavor-space position, mixture distinctiveness /
key components, good-flavor probability, and ideal ratios.

## Reproduce

```bash
python recipe_classifier.py   # leave-one-flavor-out AUC + demo scores
python denovo_recipes.py      # recipes carry good-flavor P and classifier-driven ratios
```
