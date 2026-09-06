# Handoff to a chemist or molecular gastronomist

This suite is a set of **computational hypotheses**, not recipes. It is meant to hand to a
flavor chemist or a molecular gastronomist to evaluate and, if appropriate, make and taste.

## What the suite is

- `results/flavor_suite_real.json` / `.md`: ranked candidate combinations of real flavor
  molecules, each predicted (by a model trained on real human pleasantness ratings) to be
  pleasant, chemically distinct from cocoa, and passing a structural safety screen.
- Each candidate lists molecule names and **SMILES** so a chemist can identify and source
  each compound.

## How it was produced

1. Pleasantness model trained on Keller & Vosshall 2016 human panel ratings (real data).
2. Candidate combos composed to maximize predicted mean pleasantness while staying novel
   vs cocoa key odorants.
3. Every molecule passed a food-appropriate structural safety screen.

## What you must do before anyone tastes anything (critical)

The safety screen is a **structural hazard filter, not a safety clearance**. Before any
human tasting:

1. **Regulatory status.** Confirm each molecule's food status (e.g. FEMA GRAS number, FDA
   21 CFR, or EU flavoring authorization). Many flavor molecules are GRAS; some are
   restricted or have use-level limits. The screen does not check this.
2. **Purity and source.** Use food-grade material from a reputable supplier with a CoA.
3. **Use levels.** Pleasantness and safety are dose-dependent. Determine safe and
   sensible concentrations; the model says nothing about dose.
4. **Allergens / individual sensitivity.** Check for known allergens and sensitizers.
5. **Mixture effects.** The ranking assumes additive pleasantness. Real mixtures are not
   additive (suppression/synergy), so treat the blend as a starting point to adjust by
   nose and palate, not a finished formula.

## What would make this stronger (feedback welcome)

- Real mixture pleasantness data to train the learned mixture model.
- A trained flavor chemist's read on which candidates are plausible vs chemically odd.
- Bench evaluation of the top few candidates (smell first, well below tasting levels).

Contact via the repository. Nothing in this repo authorizes ingestion; proceed only under
appropriate professional and regulatory review.
