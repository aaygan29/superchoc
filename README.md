# superchoc

**A computational instrument for designing novel, safe, pleasant flavor combinations, built
toward Conjecture 4 ("Superchocolate exists in flavor space").**

The conjecture and framing are **not the author's own**: they come from a recent essay on
biological challenges for AI (2026). The claim, briefly: chocolate is a very high-dimensional
flavor (hundreds of odorants across ~30 taste and ~400 odor receptors), flavor space is vast and
mostly unexplored, so exceptional unexperienced flavors likely exist; and because odor receptors
are GPCRs (the target of ~36% of approved drugs), solving flavor design may advance drug design.
**The conjecture and framing are the essay's; the implementation and analysis here are the
author's.**

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

**What the tool actually proposes.** The unit of proposal is a **flavor**, i.e. a *combination*
of molecules (a recipe with proportions), not a claim that any single molecule is a pleasant
new odorant. Per-molecule pleasantness predictions are *building-block* signals used to assemble
and rank combinations; the novelty and good-taste claims live at the recipe level. De-novo
molecules in particular are structural building blocks with predicted (extrapolated) properties,
not validated pleasant compounds on their own.

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
6. **Check** an elementary boundedness property of the additive metric in Lean (`flavor_math` /
   `FlavorMath.lean`; a definitional list-maximum bound, not a deep result) and **validate** every
   model on public data with honest controls, including similarity-aware splits and held-out
   real mixture transfer.

---

## Main results (all reproducible; see REPLICATION.md)

| Result | Value | Where |
|---|---|---|
| Structure -> human pleasantness (Keller 2016) | random 5-fold CV Spearman **0.50**, R2 0.26; Butina-cluster split **0.41**/0.22; scaffold split **0.30** (the extrapolation number) | `goodness_real.py`, audit `results/newslate/audit_claims.json` |
| Flavor-space *position* -> pleasantness (goodness field) | Spearman **0.49** (~ matches ECFP 0.50) | `flavor_field.py` |
| Flavor space is hyperbolic (Poincare vs Euclidean, 3D) | Spearman **0.59 vs 0.27** | `flavor_geometry.py` |
| Flavor-space tree-likeness (Gromov delta vs shuffled) | real **0.0188 < 0.0205** | `flavor_geometry.py` |
| Mixture perceptual-distance vs human ratings (Snitz 2013) | Spearman **-0.49** on 360 pairs | `mixtures.py` |
| Mixture perceptual-distance vs human ratings (Ravia 2020) | Spearman **-0.245** on 195 pairs | `mixtures.py` |
| Learned non-additive mixture vs mean-pool (synthetic oracle) | Spearman **0.82 vs 0.25** — but this gap is *built into* the oracle (an additive oracle gives 0.84 vs 0.82); on clean held-out real data (Snitz<->Ravia transfer) the set-vs-meanpool advantage is **within the CI of zero** (e.g. +0.03 [-0.06, 0.12]). The non-additive advantage is **not supported on real data.** | `mixture_model.py`, audit `results/newslate/real_mixture_eval.json` |
| Recipe-level good-flavor classifier (leave-one-flavor-out, 23 pleasant chemotype-diverse flavors) | ROC-AUC **0.72** (RF 0.73); more robust than the 6-flavor 0.69, and savory/pungent classes were correctly excluded | `recipe_classifier.py` |
| Significance of headline results (bootstrap CI + permutation p + FDR) | all 3 mixture/field results **significant** (FDR < 0.01) | `significance.py` |
| Molecule -> odorant-receptor signal (Mainland 2015) | **1.57x** chance = 9.1% NN-shared-receptor vs 5.8% chance on only **n=22 pairs** (2 hits, one mutual pair; binomial p=0.37, CI reaches chance). A weak, data-limited signal: **diagnostic only, dropped as a usable receptor-in-J term** (M2OR-vs-pleasantness Spearman 0.14, p=0.42) | `receptors.py`, audit `audit_claims.json` |
| Molecule -> receptor activation model (M2OR, leave-molecule-out) | ROC-AUC **0.89** (92 mols x 375 human ORs) | `m2or_receptor.py` |
| Structure-based receptor binding (Boltz-2, held-out swap) | cognate vs non-cognate AUC **0.875** | `receptor_binding.py` |
| Combo-pleasantness benchmark (real Keller labels + literature pooling) | set **0.70** vs additive **0.67** — a 0.03 gap, within noise; no direct human combo-pleasantness data exists | `combo_pleasantness.py` |
| Composition -> blend pleasantness | Spearman **0.42** | `composition.py` |
| Food-safety toxicity screen | 380/420 pass; benzene/formaldehyde flagged | `safety.py` |
| De-novo molecules generated | 434 safe & novel; **top-12 absent from PubChem** (database novelty only; SA score ~2.2 does not establish a synthetic route, and PubChem absence is not synthetic feasibility) | `lego_assembly.py`, audit `generator_baselines.json` |
| Guided vs unguided generation (honesty check) | guided top-40 scores +12.4 under the ranking model but only **+1.2** under an independent descriptor GBM (predictor corr 0.27): most of the guided gain is **selection on the predictor itself** | audit `generator_baselines.json` |
| 3D geometry gate (de-novo validity) | **10/10** real pass, **4/4** invalid rejected | `geometry_gate.py` |
| Metric best-part bound (Lean) | Lean exits 0, but the theorems formalize only the elementary list-maximum bound over `Nat` (no weights, no convexity, no real-valued T(R)): a **definitional footnote, not a results-table result** | `FlavorMath.lean` |
| Online active search vs random screening | +0.074, 25/25 seeds (synthetic) | `methods/active_flavor_search/` |

An honest tension the tool measures rather than hides: **chemical novelty is not perceptual
novelty**, and pushing novelty lowers the good-flavor score. The `analyzer` exposes this per
recipe (novelty vs good-flavor probability) instead of asserting a recipe is both maximally novel
and maximally good.

---

## Specification and self-audit of the generative objective

The generative core of the pipeline is a multi-objective search, so it is specified and audited
in full (`methods/real_flavor/newslate/`, findings in `results/NEWSLATE_findings.md`). We report
the results honestly, including where they refute the pipeline's own earlier claims.

1. **The search objective, fully specified.** The objective is defined in `newslate/J_SPEC.md` and
   implemented in `newslate/jspec.py` (verified to reproduce the legacy generator exactly, then
   extended) with a non-additive mixture term `M`:
   `J = w_t·T + w_c·C + w_n·N + w_mix·M`, weights `(1.0, 0.15, 0.4, 0.4)`, equal proportions,
   stochastic 1-swap hill climbing (5 restarts × 500 proposals, hard novelty floor N≥0.5). `M` is
   trained on human *similarity* ratings (Snitz/Ravia), not pleasantness, so it models perceptual
   distinctiveness, not goodness. A 2⁴ weight ablation (110 runs,
   `methods/real_flavor/results/newslate/generation_analysis.json`, figure
   `figures/fig1_weight_ablation.pdf`) shows `M` is the **least influential term**: its main effect
   on every output metric is within the CI of zero, dropping it changes the generated set by a
   Jaccard of 0.61 (noise floor 0.49), and the learned `M` is itself nearly additive in practice
   (M minus additive-aggregate ≈ 0.005). Taste and composition drive the recipes; the non-additive
   term does no real work on this dataset.
2. **The non-additive advantage does not survive real data.** The 0.82-vs-0.25 mixture result is a
   property of a synthetic oracle built to be non-additive (an additive oracle inverts it). On
   clean held-out real mixtures (Snitz↔Ravia transfer, molecule-disjoint) the set-vs-meanpool
   advantage falls within the CI of zero, and a size-only model captures much of the signal. The
   claim that the learned mixture model beats additive pooling is therefore **not supported on real
   data** and is withdrawn.
3. **The receptor layer is a weak signal and is not scored into the objective.** 1.57× is
   9.1% vs 5.8% on 22 pairs (p=0.37); the measured receptor-to-pleasantness map is null
   (Spearman 0.14, p=0.42). Kept as diagnostic context, not a term in `J`.
4. **The Lean result is definitional.** It checks an elementary list-maximum bound, not the
   weighted/convex tastiness estimate. Reported as a footnote, not a headline result.

**Known gaps still open.** (a) *Concentration.* Olfactory response is concentration-dependent;
in the Keller data, pleasantness shifts ~8 points between dilutions and flips sign for ~30% of
molecules, yet `J` uses equal proportions and no dose. A concentration-aware mixture model needs
dose-response labels per component, which the two Keller dilutions cannot fit. (b) *Alternative
generators.* The de-novo candidates are compared against unguided and re-seeded BRICS baselines
(`generator_baselines.py`), but not against a different generation method class (e.g. a learned
generative model); PubChem absence and an SA score of ~2.2 establish database novelty and nominal
ease, not a synthetic route. Both are honest limits, not claimed as solved.

Scope: this is an independent, self-funded project on public data, with no laboratory, no
synthesis capacity, and no human sensory panel. Physically constructing these molecules and
measuring real human mixture pleasantness, the validation that would settle the central claim, is
out of reach here; the outputs remain ranked hypotheses for a wet-lab group, not results.

---

## Repository map

```
superchoc/
  README.md            <- you are here
  REPLICATION.md       <- exact steps + methodology to reproduce everything
  ROADMAP.md           <- the conjecture's 3 success criteria, decomposed
  requirements.txt     <- Python deps
  paper/               <- superchoc_v1_older_article-class.{tex,pdf}: an OLDER article-class draft,
                          superseded; the honest current state is this README + newslate/ audit
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
| `m2or_receptor.py` | learned molecule -> receptor activation (M2OR, LOMO AUC 0.89) | |
| `receptor_binding.py` | structure-based molecule<->receptor co-fold (Boltz-2) | RECEPTOR_BINDING |
| `combo_pleasantness.py` | mixture-pleasantness benchmark (real labels + literature pooling) | |

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
| `significance.py` | bootstrap CIs + permutation p-values + FDR for the headline results |
| `reference_flavors.py` | 23 pleasant, chemotype-diverse reference flavors (literature key-odorant sets) |
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
