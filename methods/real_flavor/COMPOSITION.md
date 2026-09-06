# Compositional statistics and statistically-composed recipes

Which chemical classes make a food taste good, in what proportion, learned from real data
and used to shape new recipes.

## Class pleasantness (real, from Keller 2016)

Each molecule is classified into functional classes (RDKit SMARTS). Mean human pleasantness
of molecules carrying each class, vs the overall mean (45.9):

| favored (positive lift) | lift | | unpleasant (negative lift) | lift |
|---|---|---|---|---|
| lactone | +9.3 | | sulfur | -18.1 |
| ether | +5.1 | | pyrazine | -10.9 |
| ester | +5.0 | | carboxylic acid | -8.4 |
| aromatic | +4.1 | | furan | -8.0 |
| phenol | +3.5 | | amine | -1.7 |

This matches flavor chemistry: esters/lactones (fruity, creamy, sweet) read pleasant;
sulfur/acids/amines (pungent, sour, fishy) read unpleasant.

## Mathematical backbone (Lean-checked)

`FlavorMath.lean` theorem `composition_bounded_by_best_class`: under a linear per-class
score, no composition can score above the best single class (weighted average <= max). So
tuning composition alone cannot beat the best class; exceeding it needs non-additive
interactions (the learned mixture model). Machine-checked, `lean FlavorMath.lean` exit 0.

## Empirical validation

Composition score (favor - avoid class balance) predicts the true mean pleasantness of
random blends at **Spearman 0.42** - a real, moderate statistical signal that composition
carries tastiness information, independent of the per-molecule model.

## Statistically-composed de-novo recipes (`denovo_recipes.py`)

Combines the composition target + goodness model + chemical novelty + safety, over a pool of
known-safe AND de-novo lego molecules. Objective:
`J = w_taste*pleasantness + w_comp*composition_score + w_novel*novelty`.

Result (`results/denovo_recipes.json`): 5 recipes, predicted pleasantness 63-66, composition
dominated by the favored ester/ether/aromatic classes, chemical novelty 0.63-0.67, and
**6-8 of 10 components are de-novo molecules** (new structures from the lego pipeline). These
are the program's best current "new flavor" hypotheses for a chemist/gastronomist, subject to
the safety and validation caveats in HANDOFF.md.

## Reproduce

```bash
python composition.py        # class pleasantness stats + target
python denovo_recipes.py     # composition-guided recipes over known + de-novo molecules
lean FlavorMath.lean         # verify the composition bound
```
