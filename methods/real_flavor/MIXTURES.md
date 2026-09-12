# Combo (recipe) science: `mixtures.py`

The conjecture is about combos, not molecules: a new *recipe* of known molecules (vanillin +
mint -> creamy peppermint) can be novel and good even when its parts are ordinary. This module
adds the mixture-level machinery, grounded in flavor-chemistry literature and validated on real
human data where possible.

## Externally validated

**Mixture perceptual-distance metric (Snitz 2013 / Ravia 2020).** A mixture is represented by
the (proportion-weightable) mean of z-scored physicochemical descriptors; perceptual distance is
the angle in radians between two mixture vectors. Validated against Snitz's **360 human-rated
mixture-similarity pairs**: Spearman **-0.49**, Pearson **-0.53** (correct sign: larger angle =
less similar). This is below the paper's ~0.7 because we use RDKit descriptors as a documented
substitute for the original Dragon feature set; we report our reimplementation's number and do
not tune on the validation set. `validate_snitz()` reproduces it.

**Scheffe simplex mixture-design + response-surface ratio optimization.** Exact math (no
perceptual claim) for optimizing component *proportions* under the sum-to-one constraint. A
degree-2 simplex-lattice design feeds a quadratic Scheffe fit; the fitted surface is searched
over the simplex. Important honest result: an ADDITIVE response optimizes to a single component
(the `FlavorMath.lean` best-part bound), so `optimize_ratios` reports `interior_optimum`: it is
`True` only when the response carries genuine interaction. Verified on a toy interaction
response (55/45 split) and observed to collapse to a vertex on purely additive objectives.

## Optimal levels (machinery only, no invented data)

`odor_activity_value` / `oav_weights` implement the flavor-chemistry standard (OAV =
concentration / detection threshold; contribute only above threshold, log-compressed for
hypoadditivity). Measured thresholds must be supplied; we deliberately ship no fabricated
threshold values.

## Hypothesis layer (labeled)

- `omission_test`: drop each component, measure the predicted-percept shift (in-silico
  aroma-recombination omission) to find key/emergent components. In the pipeline this uses the
  Snitz-validated representation, so the per-recipe `key_components` are grounded.
- `synergy_score`: predicted combo goodness minus the additive expectation (Bliss-style);
  positive = hyperadditive/emergent. This needs a validated non-additive combo-goodness model,
  which does not yet exist for real molecules, so it is exposed as a function and not baked into
  recipes as a measured number.

## Guardrail

`olfactory_white_risk`: Weiss & Sobel 2012 showed ~30+ equal components converge to an
indistinct "olfactory white"; Laing showed humans resolve at most ~3-4. More components is not
more novel; this penalizes over-complex recipes.

## Wired into the pipeline

`denovo_recipes.compose()` attaches a `mixture` field to every recipe (perceptual
distinctiveness vs each known reference flavor + nearest flavor, olfactory-white risk, omission
key-components) and derives `optimal_levels` for the top recipe via Scheffe optimization, with
the honest interior-vs-additive reading. Artifact: `results/denovo_recipes.json`.

## Reproduce

```bash
python mixtures.py          # Snitz validation + machinery demos
python denovo_recipes.py    # recipes now carry the combo analysis + optimal levels
```

Refs: Snitz 2013 (PLoS Comput Biol) doi:10.1371/journal.pcbi.1003184; Ravia 2020 (Nature)
doi:10.1038/s41586-020-2891-7; Weiss & Sobel 2012 (PNAS) doi:10.1073/pnas.1208110109;
Grosch 2001 (Chem Senses 26:533, OAV/omission); Scheffe 1958 (mixture designs).
