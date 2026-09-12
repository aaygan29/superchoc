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
2. **Low-dimensional map, Euclidean vs hyperbolic.** A flat linear PCA embedding preserves
   perceptual (Jaccard) distance only weakly (**Spearman ~0.24-0.27**, flat across 3-30 dims).
   A **Poincare-ball (hyperbolic) embedding** fit to the same distances (torch, autograd,
   `poincare_faithfulness`) does much better and wins at every dimension, replicating
   Sharpee et al. on our data:

   | dim | hyperbolic Spearman | Euclidean Spearman | improvement |
   |---|---|---|---|
   | 2 | 0.566 | 0.260 | +0.31 |
   | 3 | **0.592** | 0.265 | +0.33 |
   | 5 | 0.651 | 0.277 | +0.37 |

   The 3D hyperbolic fit ($\approx$0.59) matches the paper's finding that a 3D hyperbolic
   space fits odor perception. The PCA coordinate is kept only as a rough visual map; the
   `locate` evaluation below uses full perceptual distance, not the lossy linear map.
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

## How it plugs into the pipeline (now wired)

`denovo_recipes.compose()` calls `locate()` on every generated recipe (via `_locate_recipes`,
guarded so a geometry failure never breaks generation) and attaches a `flavor_space` field:
map coordinate, `nearest_known_flavors`, `novelty_percentile` (vs same-size random blends), and
an interpretation. So each proposed recipe now carries a hypothesis for where it lands in flavor
space before any panel tastes it. Artifact: `results/denovo_recipes.json`.

**Honest finding this surfaced.** The de-novo recipes are highly novel *chemically*
(Tanimoto-based novelty 0.6-0.7, 5-7 of 10 components de-novo) but sit at LOW *perceptual*
novelty (percentile 0.0-0.22, "among known flavors"): their nearest neighbours are familiar
esters/salicylates. Chemical novelty is not perceptual novelty. The new molecules smell like
known compounds, so the blends land in populated flavor-space regions rather than empty ones.
For a "make it taste good" goal that may be fine (or even desirable); for "reach a genuinely
unexperienced flavor" it says the current generator is not yet pushing into sparse regions. The
geometry layer makes that tension measurable instead of assumed.

Full hyperbolic (Poincare/Lorentz) placement of the located blend, following Sharpee, is the
natural next step and would sharpen the map coordinate.

## Reproduce

```bash
python flavor_geometry.py     # hyperbolicity test + embedding faithfulness + locate() demos
```
Result artifact: `results/flavor_geometry.json`.
