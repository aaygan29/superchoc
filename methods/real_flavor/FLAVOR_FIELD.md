# Goodness field over flavor space (`flavor_field.py`)

Turns the descriptive flavor-space geometry into a predictive + generative tool.

## Idea
Flavor space is a low-dimensional metric space (Kurtz 2000), empirically hyperbolic
(Sharpee 2018; replicated in `flavor_geometry.py`). "Good flavor" is a smooth SCALAR FIELD over
that space, not a direction: fruity/vanilla/lactonic regions are pleasant, sulfur/acid regions
are not. We fit that field and use it to (1) predict a molecule/blend's pleasantness from WHERE
it lands, and (2) find high-field + SPARSE regions as superchocolate targets. Because the space
is hyperbolic (volume grows exponentially with radius), there is exponentially much unexplored
room, which is the precise statement of "flavor space is vast and mostly unexplored."

## Method
Coordinates are the physicochemical mixture-space vector (`mixtures.MixtureSpace`), the same
space whose distance is externally validated against human mixture similarity (Snitz -0.49,
Ravia -0.245). The field is RBF kernel-ridge regression from those coordinates to real Keller
pleasantness, with hyperparameters chosen by 5-fold CV (proper model selection).

## Validation (`results/flavor_field.json`)
| predictor | 5-fold CV Spearman |
|---|---|
| goodness field (flavor-space POSITION only, 13-dim) | **0.49** (R^2 0.11) |
| ECFP structure model (2048-bit) baseline | 0.50 |

The headline: predicting pleasantness from *where a molecule lands* in the validated
flavor-space metric is nearly as good as predicting from its full molecular fingerprint. This is
direct evidence that pleasantness is a smooth field over the perceptual manifold, and it makes
the geometry predictive, not just descriptive. It is honestly a rank predictor (modest R^2), so
use it for relative goodness and region-finding, not calibrated absolute scores.

Field goodness of known flavors orders sensibly: rose 59 > vanilla 54 > strawberry 50 >
banana 49 > chocolate 46 > coffee 40 (sweet/floral/fruity land higher than roasty/bitter).

## Use in the pipeline
`FlavorField.field_value(smiles)` scores any blend by position; `superchoc_score` combines field
goodness with sparsity (distance to known molecule clouds) to rank high-goodness UNEXPLORED
targets. The analyzer attaches `flavor_field` (goodness + sparsity + superchoc_score) to every
recipe, which quantifies the novelty-vs-goodness tension per recipe.

## Reproduce
```bash
python flavor_field.py   # CV vs ECFP baseline + field goodness of known flavors
```
