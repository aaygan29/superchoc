# superchoc

**A computational instrument for designing novel, safe, pleasant flavor combinations, built
toward Jake Wintermute's Conjecture 4 ("Superchocolate exists in flavor space").**

The conjecture and framing are **Jake Wintermute's**, from *"Strange and Marvelous
Challenges for Biological AI"* (Sept 2026), via American Wetware
([Substack](https://americanwetware.substack.com/); author
[LinkedIn](https://www.linkedin.com/in/jake-wintermute/)). His claim, briefly: chocolate is a
very high-dimensional flavor (hundreds of odorants across ~30 taste and ~400 odor receptors),
flavor space is vast and mostly unexplored, so exceptional unexperienced flavors likely exist;
and because odor receptors are GPCRs (the target of ~36% of approved drugs), solving flavor
design may advance drug design. **The ideas are his; the implementation and analysis here are
the author's.** Read his article for the full argument.

---

## What this repository is

A working pipeline that follows Wintermute's molecule -> receptor -> percept chain and turns
it into flavor design:

1. **Predict** how pleasant a molecule is from its structure (real human-panel labels).
2. **Explain** it through odorant-receptor activation and odor descriptors.
3. **Generate** new molecules (de novo) and blend them into novel, safety-screened recipes
   with proportions and instructions, while **minimizing toxicity** and **maximizing** predicted
   pleasantness and novelty.
4. **Prove** the properties of the scoring metric (Lean) and **validate** the models on real
   data and with honest controls.

Nothing here is a wet-lab result. Everything real is validated on public data or with stated
controls; the generated flavors are **computational hypotheses for a chemist or molecular
gastronomist**, not recipes cleared to eat (see the handoff and safety notes).

---

## Main results (all reproducible, see REPLICATION.md)

| Result | Value | Where |
|---|---|---|
| Structure -> human pleasantness (real, Keller 2016) | 5-fold CV Spearman **0.50**, R2 0.26 | `methods/real_flavor/goodness_real.py` |
| Learned mixture model vs mean-pooling (non-additive) | Spearman **0.82 vs 0.25** | `mixture_model.py` |
| Molecule -> odorant-receptor signal (Mainland 2015) | **1.57x** chance | `receptors.py` |
| Composition -> blend pleasantness | Spearman **0.42** | `composition.py` |
| Food-safety toxicity screen | 380/420 pass; benzene/formaldehyde/epoxide flagged | `safety.py` |
| De-novo molecules generated | 434 safe & novel; **top-12 all absent from PubChem** | `lego_assembly.py` |
| 3D geometry gate (de-novo validity) | **10/10** real molecules pass, **4/4** invalid rejected; library 100% valid | `geometry_gate.py` |
| Structure-based receptor binding (Boltz-2 co-fold, held-out specificity swap) | **separates cognate from non-cognate: AUC 0.875, 2/2 receptors** | `receptor_binding.py` |
| Metric properties (best-part bound, boundedness) | **machine-checked in Lean** (exit 0) | `FlavorMath.lean` |
| Flavor space is hyperbolic (Poincaré vs Euclidean, 3D) | Spearman **0.59 vs 0.27** (+0.33) | `flavor_geometry.py` |
| Flavor-space tree-likeness (Gromov δ vs shuffled) | real **0.0188 < 0.0205** shuffled | `flavor_geometry.py` |
| Locate-a-blend novelty layer | discriminates common (0.003) vs de-novo (~0.40) blends | `flavor_geometry.py` |
| Online active search vs random screening | +0.074, 25/25 seeds (synthetic) | `methods/active_flavor_search/` |
| Final novel recipes | 5 blends, 6-8/10 de-novo components, novelty 0.63-0.67 | `denovo_recipes.py` |

---

## Repository map

```
superchoc/
  README.md            <- you are here
  REPLICATION.md       <- exact steps + methodology to reproduce everything
  ROADMAP.md           <- Wintermute's 3 success criteria, decomposed
  requirements.txt     <- Python deps
  paper/               <- superchoc_v1_older_article-class.{tex,pdf}: an OLDER article-class
                          draft. The current NeurIPS-template submission (with the geometry
                          results) lives in the submission bundle, not in this repo.
  docs/                <- background and honest framing
    LITERATURE.md         literature synthesis + technique library
    GROUND_TRUTH.md       the public datasets/benchmarks that anchor each claim
    CROSS_DISCIPLINE.md   how chemistry/receptor-biology/neuro/neuropsych integrate
    SYNTHESIS.md          feasibility assessment of bridging the criteria
  data/                <- dataset loaders + live online ingestion (no fabricated data)
  methods/
    active_flavor_search/   online (sequential) design: which molecule to taste next
    goodness_model/         goodness model + generator on synthetic ground truth (Goodhart control)
    real_flavor/            THE MAIN PIPELINE (real data): everything below
```

### `methods/real_flavor/` (the main pipeline)

| file | role | doc |
|---|---|---|
| `real_data.py` | real molecules + real pleasantness (Keller 2016 via Pyrfume) | |
| `featurize.py` | SMILES -> ECFP4 | |
| `goodness_real.py` | pleasantness model (Criterion 1) | README |
| `flavor_profile.py` | odor descriptors (Leffingwell) | CROSS_DISCIPLINE |
| `receptors.py` | molecule -> odorant-receptor activation (Mainland 2015) | |
| `receptor_binding.py` | structure-based molecule<->receptor co-fold (Boltz-2) | RECEPTOR_BINDING |
| `geometry_gate.py` | 3D-validity gate for de-novo molecules (ETKDG + MMFF/UFF) | GEOMETRY_GATE |
| `neuro_reward.py` | reward/mood molecules (caffeine, PEA, ...) | |
| `safety.py` | food-appropriate toxicity screen | HANDOFF |
| `mixture_model.py` | learned non-additive mixture model | |
| `composition.py` | chemical-class composition statistics | COMPOSITION |
| `flavor_math.py` + `FlavorMath.lean` | tastiness metric + Lean-checked theorems | MATH |
| `flavor_geometry.py` | flavor-space geometry: hyperbolicity test + locate-a-blend | GEOMETRY |
| `lego_assembly.py` | de-novo molecule construction + PubChem novelty | LEGO |
| `recipes.py` / `flavor_designer.py` / `denovo_recipes.py` | recipe generators | |
| `results/` | committed JSON/markdown for every result above | |

---

## Quick start

```bash
pip install -r requirements.txt          # numpy, scikit-learn, scipy, rdkit, pyrfume
cd methods/real_flavor
python validate_real.py                  # real pleasantness CV + mixture + safety gate + novelty
python denovo_recipes.py                 # the final composition-guided novel recipes
python lego_assembly.py                  # de-novo molecules + 3D geometry gate + PubChem novelty
python geometry_gate.py                  # 3D-validity gate: known-good vs invalid vs de-novo
python validate_receptor_binding.py      # Boltz-2 receptor-specificity swap test (needs credits)
lean FlavorMath.lean                     # machine-check the metric theorems (needs Lean 4)
```

See **REPLICATION.md** for the full, step-by-step guide and how to go from these
computational hypotheses to a real, panel-tested flavor.

---

## Honesty and safety

- Real claims are validated on public data with stated controls; generated flavors are ranked
  **hypotheses**, not proven-tasty or safe-to-eat.
- The safety screen is a structural **hazard filter, not a GRAS/regulatory clearance**. De-novo
  molecules have no safety data at all.
- "Novel" means distinct from known/public flavors and (for structures) absent from PubChem; it
  does **not** prove absence from proprietary formulas or from nature.
- See `methods/real_flavor/HANDOFF.md` before anyone makes or smells anything.
