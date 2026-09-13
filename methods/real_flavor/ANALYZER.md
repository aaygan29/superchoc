# SuperchocAnalyzer: the overarching analyzer (`analyzer.py`)

One entry point that synthesizes the whole pipeline. It takes chemical structures, per-molecule
predictions, known reference flavors, and the geometric flavor-space representation, and returns
internally-evaluated predictions for NEW flavor combos that are novel and theorized to taste
good, each with a plain-language description of the intended taste.

## What it unifies
| sub-model | contribution |
|---|---|
| `goodness_real` | per-molecule human pleasantness (real Keller labels) |
| `safety` | food-appropriate toxicity gate (hard filter) |
| `lego_assembly` | de-novo molecules, for structural novelty |
| `flavor_geometry` | hyperbolic flavor-space position + perceptual novelty percentile |
| `mixtures` | Snitz/Ravia-validated mixture perceptual distance + Scheffe ratio optimizer |
| `recipe_classifier` | recipe-level P(good flavor), leave-one-flavor-out AUC 0.69 |
| `taste_profile` | five basic tastes + chef-descriptor flavor regions + written description |

## API
- `SuperchocAnalyzer(allow_denovo=True)` loads every sub-model once (memoized featurization
  keeps it fast).
- `analyze_structure(smiles)` -> per-molecule pleasantness, safety, classes, flavor-space
  novelty percentile, nearest known flavors.
- `design(n_recipes, n_components, novelty_weight, strength_weight)` -> ranked recipes. The
  search maximizes a light objective (classifier P(good) + perceptual novelty); the expensive
  taste profile is computed only when packaging the finalists. Ratios are then optimized against
  the classifier (a non-additive response).
- `report()` -> the ranked recipes plus the validation record, written to
  `results/superchoc_analysis.json`.

Each recipe package carries: components (name, SMILES, source, proportion, pleasantness,
classes), good-flavor probability, chemical + perceptual novelty, profile strength, the five
basic tastes, dominant taste, top chef-descriptors, flavor region, nearest known flavor, optimal
ratios, and a natural-language `taste_description`.

## Honest behavior
- **Novelty vs goodness is a real tension.** With `novelty_weight` high and de-novo molecules
  allowed, the good-flavor probability drops (observed 0.20-0.46) and profiles skew
  herbal/bitter/animal, because the classifier (trained on real flavors) does not recognize
  de-novo-heavy blends as flavor-like. Lowering `novelty_weight` (or `validated_recipe.py`, which
  uses known-safe molecules) reaches the high-confidence end (P(good) ~0.84). The knob spans the
  tradeoff honestly rather than hiding it.
- **Internal evaluation only.** Every score comes from a validated or externally-grounded model
  (classifier LOFO AUC 0.69; Snitz -0.49 / Ravia -0.245 mixture distance; Keller pleasantness)
  plus a safety gate. Nothing is human-tasted; outputs are ranked hypotheses for a chemist or
  chef (HANDOFF.md).

## Companion: `validated_recipe.py`
Produces a single high-confidence recipe from known-safe molecules (P(good) ~0.84, strong
focused profile), with the full taste categorization and optimized ratios. Use it when you want
one solid, characterizable recipe rather than a novelty-leaning ranked set.

## Reproduce
```bash
python analyzer.py           # ranked novel good-taste recipes + taste descriptions
python validated_recipe.py   # one high-confidence strong-profile recipe
python taste_profile.py      # five-taste + flavor-region characterization demo
```
