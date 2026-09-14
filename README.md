# superchoc

**A computational instrument for designing novel, safe, pleasant flavor combinations, built
toward Jake Wintermute's Conjecture 4 ("Superchocolate exists in flavor space").**

The conjecture and framing are **Jake Wintermute's**, from *"Strange and Marvelous Challenges
for Biological AI"* (Sept 2026), via American Wetware
([Substack](https://americanwetware.substack.com/); author
[LinkedIn](https://www.linkedin.com/in/jake-wintermute/)). His claim, briefly: chocolate is a
very high-dimensional flavor (hundreds of odorants across ~30 taste and ~400 odor receptors),
flavor space is vast and mostly unexplored, so exceptional unexperienced flavors likely exist;
and because odor receptors are GPCRs (the target of ~36% of approved drugs), solving flavor
design may advance drug design. **The ideas are his; the implementation and analysis here are
the author's.** Read his article for the full argument.

---

## What this is

A working pipeline that follows the biological chain **molecule -> receptor -> percept -> combo
-> recipe** and turns it into flavor design. It does not just find good molecules; it composes
and evaluates good *combinations* (a recipe, like chocolate, is hundreds of molecules together),
and it reports, for each proposed recipe, a written description of what the flavor should taste
like and where it sits in flavor space.

Everything is **internal, in-silico evaluation** on validated or externally-grounded models plus
a safety gate. Nothing here is a wet-lab result; the outputs are ranked **hypotheses for a
chemist or molecular gastronomist**, not recipes cleared to eat (see `HANDOFF.md`).

### Start here
```bash
pip install -r requirements.txt          # numpy, scikit-learn, scipy, rdkit, pyrfume
cd methods/real_flavor
python analyzer.py                        # THE analyzer: ranked novel good-taste recipes + a superchocolate
```
`analyzer.py` (`SuperchocAnalyzer`) is the single overarching entry point. It ingests chemical
structures, per-molecule predictions, known reference flavors, and the geometric flavor-space
representation, and returns ranked novel recipes, each with a good-flavor probability, chemical +
perceptual novelty, a five-taste profile, optimized component ratios, and a plain-language taste
description. It runs a compute-efficient cascade (cheap filters first, expensive scoring only on
survivors). `validated_recipe.py` produces one high-confidence recipe instead of a ranked set.

---

## How it works (the stages)

1. **Predict** how pleasant a molecule is from its structure (`goodness_real`, real Keller labels).
2. **Place** it in flavor space and predict pleasantness from *where it lands* (`flavor_geometry`
   + `flavor_field`): flavor space is a low-dimensional, empirically hyperbolic metric space, and
   "good" is a smooth field over it.
3. **Explain** it through odorant-receptor activation (`receptors`, `receptor_binding`) and odor
   descriptors / the five basic tastes (`flavor_profile`, `taste_profile`).
4. **Generate** new molecules de novo (`lego_assembly`), gated for 3D validity (`geometry_gate`)
   and food safety (`safety`).
5. **Compose** them into combos with validated mixture science (`mixtures`: Snitz/Ravia-validated
   perceptual distance, Scheffe ratio optimization, omission testing) and a recipe-level
   good-flavor classifier (`recipe_classifier`).
6. **Prove** the scoring metric's properties in Lean (`flavor_math` / `FlavorMath.lean`) and
   **validate** every model on public data with honest controls.

---

## Main results (all reproducible; see REPLICATION.md)

| Result | Value | Where |
|---|---|---|
| Structure -> human pleasantness (Keller 2016) | 5-fold CV Spearman **0.50**, R2 0.26 | `goodness_real.py` |
| Flavor-space *position* -> pleasantness (goodness field) | Spearman **0.49** (~ matches ECFP 0.50) | `flavor_field.py` |
| Flavor space is hyperbolic (Poincare vs Euclidean, 3D) | Spearman **0.59 vs 0.27** | `flavor_geometry.py` |
| Flavor-space tree-likeness (Gromov delta vs shuffled) | real **0.0188 < 0.0205** | `flavor_geometry.py` |
| Mixture perceptual-distance vs human ratings (Snitz 2013) | Spearman **-0.49** on 360 pairs | `mixtures.py` |
| Mixture perceptual-distance vs human ratings (Ravia 2020) | Spearman **-0.245** on 195 pairs | `mixtures.py` |
| Learned non-additive mixture vs mean-pool (synthetic oracle) | Spearman **0.82 vs 0.25** | `mixture_model.py` |
| Recipe-level good-flavor classifier (leave-one-flavor-out) | ROC-AUC **0.69** (unseen flavor) | `recipe_classifier.py` |
| Molecule -> odorant-receptor signal (Mainland 2015) | **1.57x** chance | `receptors.py` |
| Structure-based receptor binding (Boltz-2, held-out swap) | cognate vs non-cognate AUC **0.875** | `receptor_binding.py` |
| Composition -> blend pleasantness | Spearman **0.42** | `composition.py` |
| Food-safety toxicity screen | 380/420 pass; benzene/formaldehyde flagged | `safety.py` |
| De-novo molecules generated | 434 safe & novel; **top-12 absent from PubChem** | `lego_assembly.py` |
| 3D geometry gate (de-novo validity) | **10/10** real pass, **4/4** invalid rejected | `geometry_gate.py` |
| Metric best-part bound / boundedness | **machine-checked in Lean** (exit 0) | `FlavorMath.lean` |
| Online active search vs random screening | +0.074, 25/25 seeds (synthetic) | `methods/active_flavor_search/` |

An honest tension the tool measures rather than hides: **chemical novelty is not perceptual
novelty**, and pushing novelty lowers the good-flavor score. The `analyzer` exposes this per
recipe (novelty vs good-flavor probability) instead of asserting a recipe is both maximally novel
and maximally good.

---

## Repository map

```
superchoc/
  README.md            <- you are here
  REPLICATION.md       <- exact steps + methodology to reproduce everything
  ROADMAP.md           <- Wintermute's 3 success criteria, decomposed
  requirements.txt     <- Python deps
  paper/               <- superchoc_v1_older_article-class.{tex,pdf}: an OLDER article-class draft
                          (the current NeurIPS-template submission lives in the submission bundle)
  docs/                <- background + honest framing (LITERATURE, GROUND_TRUTH, CROSS_DISCIPLINE, SYNTHESIS)
  data/                <- dataset loaders + live online ingestion (no fabricated data)
  methods/
    active_flavor_search/  online sequential design: which molecule to taste next (synthetic-validated)
    goodness_model/        goodness model + generator on synthetic ground truth (Goodhart control)
    real_flavor/           THE MAIN PIPELINE (real data) -- see below
```

### `methods/real_flavor/` (the main pipeline)

**Orchestration (start here)**
| file | role | doc |
|---|---|---|
| `analyzer.py` | overarching analyzer: everything -> ranked novel good-taste recipes + descriptions | ANALYZER |
| `validated_recipe.py` | one high-confidence strong-profile recipe (full package) | ANALYZER |

**Molecule level**
| file | role | doc |
|---|---|---|
| `real_data.py` | real molecules + real pleasantness (Keller 2016 via Pyrfume) | |
| `featurize.py` | SMILES -> ECFP4 | |
| `goodness_real.py` | structure -> pleasantness model | |
| `flavor_profile.py` | odor descriptors (Leffingwell) | CROSS_DISCIPLINE |
| `safety.py` | food-appropriate toxicity screen | HANDOFF |
| `neuro_reward.py` | reward/mood molecules (caffeine, PEA, ...) | |

**Receptors**
| file | role | doc |
|---|---|---|
| `receptors.py` | molecule -> odorant-receptor activation (Mainland 2015) | |
| `receptor_binding.py` | structure-based molecule<->receptor co-fold (Boltz-2) | RECEPTOR_BINDING |

**Flavor-space math**
| file | role | doc |
|---|---|---|
| `flavor_geometry.py` | flavor-space geometry: hyperbolicity + locate-a-blend | GEOMETRY |
| `flavor_field.py` | goodness field over flavor space (predict from position; find sparse-good targets) | FLAVOR_FIELD |
| `flavor_math.py` + `FlavorMath.lean` | tastiness metric + Lean-checked theorems | MATH |

**Combos, recipes, and taste**
| file | role | doc |
|---|---|---|
| `lego_assembly.py` | de-novo molecule construction + PubChem novelty | LEGO |
| `geometry_gate.py` | 3D-validity gate for de-novo molecules (ETKDG + MMFF/UFF) | GEOMETRY_GATE |
| `mixtures.py` | combo science: validated mixture-distance, Scheffe ratio optimization, OAV, omission | MIXTURES |
| `recipe_classifier.py` | recipe-level good-flavor classifier (drives optimal ratios) | RECIPE_CLASSIFIER |
| `taste_profile.py` | five basic tastes + chef-descriptor flavor regions + steering lever | ANALYZER |
| `composition.py` | chemical-class composition statistics | COMPOSITION |
| `mixture_model.py` | learned non-additive mixture model (synthetic oracle) | |
| `denovo_recipes.py` | composition-guided recipe generator (feeds the analyzer) | |

**Validation / reproduction**
| file | role |
|---|---|
| `validate_real.py` | real pleasantness CV + mixture + safety gate + novelty |
| `validate_receptor_binding.py` | Boltz-2 receptor-specificity swap test (needs Boltz credits) |
| `suite_real.py` | end-to-end suite over the real-data pipeline |
| `test_real_flavor.py` | unit tests |
| `results/` | committed JSON/markdown artifact for every result above |

---

## Reproduce key pieces
```bash
cd methods/real_flavor
python analyzer.py                  # ranked novel good-taste recipes + superchocolate
python validated_recipe.py          # one high-confidence strong-profile recipe
python flavor_field.py              # goodness field CV vs ECFP baseline
python mixtures.py                  # Snitz + Ravia mixture-distance validation
python recipe_classifier.py         # leave-one-flavor-out AUC
python validate_real.py             # pleasantness CV + mixture + safety + novelty
lean FlavorMath.lean                # machine-check the metric theorems (needs Lean 4)
```
See **REPLICATION.md** for the full step-by-step guide.

---

## Honesty and safety

- All evaluation is internal, on validated/externally-grounded models plus a safety gate.
  Generated flavors are ranked **hypotheses**, not proven-tasty or safe-to-eat, and nothing has
  been human-tasted.
- The safety screen is a structural **hazard filter, not a GRAS/regulatory clearance**. De-novo
  molecules have no safety data at all.
- "Novel" means distinct from known/public flavors and (for structures) absent from PubChem; it
  does **not** prove absence from proprietary formulas or from nature.
- The recipe classifier learns "looks like a real flavor" (known flavors vs random blends), which
  is a useful prior, **not** a proof of deliciousness; there is no human combo-deliciousness
  ground truth yet.
- See `methods/real_flavor/HANDOFF.md` before anyone makes or smells anything.
