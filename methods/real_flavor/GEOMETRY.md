# Flavor-space geometry: where a blend lands

Answers a specific question: given a molecular blend, **where does it sit in flavor space,
and is that a known or an unexplored region?** This adds an evaluation/hypothesis layer on
top of the pleasantness/novelty scores, grounded in the mathematics of odor space.

## The mathematics of flavor space (what it means)

- Flavor/odor space is a **metric space**: distance = perceptual dissimilarity
  (Kurtz et al. 2000, [DOI](https://doi.org/10.3758/bf03212093)).
- It is empirically **low-dimensional** despite the huge dimensionality of molecular
  structure, and it is best fit by **hyperbolic (negative-curvature) geometry**, because
  odor co-occurrence in natural mixtures is hierarchical / tree-like. A 3D hyperbolic space
  fits both natural odor statistics and human descriptions (Zhou, Smith & Sharpee 2018,
  *Sci Adv*, [DOI](https://doi.org/10.1126/sciadv.aaq1458)).
- Learned Euclidean embeddings (Principal Odor Map) also make distance predict perceptual
  similarity (Lee et al. 2023).

## What the module does (`flavor_geometry.py`)

Using Leffingwell odor descriptors as the perceptual representation of ~2500 molecules:

1. **Tree-likeness / hyperbolicity test.** Normalized Gromov 4-point delta of the perceptual
   distance matrix vs a co-occurrence-shuffled control. Result: real delta **0.0188** <
   shuffled **0.0205**, i.e. flavor space is **more tree-like than the control**, consistent
   with the hyperbolic-structure literature (the effect is modest, as expected for a 113-note
   descriptor space).
2. **Low-dimensional map.** A PCA coordinate embedding. Honest finding: a flat linear 3D
   Euclidean embedding preserves perceptual (Jaccard) distance only weakly (**Spearman ~0.24**,
   flat across 3-30 dims), which is exactly why the literature turns to hyperbolic geometry.
   So the PCA coordinate is reported only as a rough map; the actual evaluation uses full
   perceptual distance.
3. **`locate(blend)`.** Places a blend by its mean descriptor profile and reports, in the
   FULL perceptual space (generalized Jaccard):
   - `nearest_known_flavors` (where it lands),
   - `novelty_percentile` vs **random blends of the same size** (a fair baseline), and
   - an interpretation (novel/unexplored region vs among known flavors).

## Sanity check (it discriminates)

| blend | novelty percentile | reading |
|---|---|---|
| common fruity esters (ethyl butyrate + hexanoate + isoamyl acetate) | 0.003 | sits among known flavors (correct: ubiquitous) |
| chocolate-like key odorants | 0.43 | moderately positioned |
| top de-novo recipe | 0.40 | moderately novel region |

The ubiquitous fruity blend is correctly flagged as **not** novel; the de-novo recipe sits in
a more sparsely populated region. This is the "where should this land" hypothesis layer:
a coordinate + nearest neighbors + an honest novelty estimate against a same-size baseline.

## How it plugs into the pipeline

`locate()` is an evaluation layer for any candidate from `recipes.py` / `denovo_recipes.py`:
attach `novelty_percentile` and `nearest_known_flavors` to each proposed blend to hypothesize
its perceptual position before a panel ever tastes it. Full hyperbolic (Poincare/Lorentz)
embedding, following Sharpee, is the natural next step and would improve the map coordinate.

## Reproduce

```bash
python flavor_geometry.py     # hyperbolicity test + embedding faithfulness + locate() demos
```
Result artifact: `results/flavor_geometry.json`.
