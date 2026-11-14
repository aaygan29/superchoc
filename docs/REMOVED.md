# What was removed, and why

The first version of this project (tag `v1-archive`, which still has every file below and
can reproduce every number) built a broad pipeline. Its own self-audit
(`methods/real_flavor/results/NEWSLATE_findings.md`) and a second module-by-module audit
found that most headline claims did not hold. Those modules were deleted rather than left in
a repo whose claims now need to be defensible. Nothing is hidden: check out `v1-archive`
to inspect them.

| Removed | Claim it supported | Why it was removed |
|---|---|---|
| `flavor_geometry.py`, `GEOMETRY.md` | flavor space is hyperbolic | Poincaré embedding fitted by MSE was compared with PCA, which does not minimise distance error, so the comparison was unfair; Gromov δ 0.0188 vs 0.0205 is one seed, no CI. |
| `flavor_field.py` | pleasantness predicted from "position" | Position was 13 RDKit descriptors, not the geometry; hyperparameters chosen on the test folds; a weaker duplicate of the ECFP model. |
| `receptors.py` | molecule → receptor signal, 1.57× chance | 2 hits of 22 pairs, p = 0.37. Null. |
| `receptor_binding.py`, `validate_receptor_binding.py` | Boltz-2 cognate AUC 0.875 | 8 co-folds (2 receptors × 4 ligands); musk vs acid confounded with ligand size. An anecdote, not a validation. |
| `mixture_model.py` | non-additive mixtures beat additive (0.82 vs 0.25) | The gap was built into a synthetic oracle; on real held-out mixtures it vanished. Withdrawn by the self-audit. |
| `combo_pleasantness.py` | set 0.70 vs additive 0.67 | Target was semi-synthetic (invented pooling law plus noise): circular. |
| `composition.py`, `COMPOSITION.md` | composition → blend pleasantness 0.42 | No code or result file produces 0.42; statistics fitted in-sample. |
| `recipe_classifier.py` | recipe-level "good flavor" AUC 0.72 | Reference flavors share molecules, so held-out flavors leak; negatives are random blends, so it learns "looks like a flavor", not liking; logistic regression scored 0.37. |
| `taste_profile.py` | five basic tastes from odor descriptors | Unvalidated heuristic. |
| `lego_assembly.py`, `geometry_gate.py`, `denovo_recipes.py` | de-novo flavor molecules | Untastable without synthesis and toxicology; guided gain was mostly selection on the predictor itself. Out of scope for a food-safe tasting experiment. |
| `analyzer.py`, `validated_recipe.py`, `suite_real.py`, `validate_real.py` | ranked "validated" recipes | Orchestrated the modules above; the "validated" flag rested on synthetic and circular checks. |
| `neuro_reward.py` | reward/mood molecules | Stimulants have no place in a tasting study; no claim was tested. |
| `flavor_math.py`, `FlavorMath.lean`, `MATH.md` | Lean-checked tastiness theorem | The theorem is the definition of a list maximum over naturals. |
| `methods/goodness_model/`, `methods/active_flavor_search/` | generator and active search | Synthetic ground truth only. |
| `methods/real_flavor/newslate/` scripts | the self-audit | Imported the removed modules. The findings and their result JSON are kept; the scripts run at `v1-archive`. |
| `data/loaders/`, `data/online_ingest.py` | data ingestion | Every loader raised `NotImplementedError`; ingest was unused. |
| `paper/` | v1 draft | Superseded; predates the retractions. |

Kept from v1: `real_data.py`, `featurize.py`, `goodness_real.py` (Keller pleasantness),
`mixtures.py` (Snitz/Ravia mixture distance), `flavor_profile.py` (Leffingwell descriptors),
`m2or_receptor.py` (exploratory; not used in any claim), `safety.py` (a structural hazard
filter, not a food-safety clearance), `reference_flavors.py`, `significance.py`, and the
self-audit record.
