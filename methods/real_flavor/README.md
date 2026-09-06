# Real-molecule flavor pipeline

The real-data version of the goodness model + generator: real molecules, real human-panel
labels, a food-appropriate safety screen, a learned mixture model, and a suite of novel
candidate flavor combinations for a lab or molecular gastronomist to test.

Everything here uses real inputs. The only synthetic component is the controlled oracle
used to *validate the learned mixture model* (there is no large public dataset of mixture
pleasantness to train on yet).

## Data (real, via Pyrfume)

`real_data.py` pulls **Keller & Vosshall 2016** (the DREAM dataset) through Pyrfume: ~420
monomolecular odorants with canonical SMILES and a human panel rating of "HOW PLEASANT IS
THE SMELL?" (0-100) at the standard 1/1000 dilution. Most pleasant come out as vanillin,
ethyl vanillin, ethyl esters, carvone; least pleasant are sulfur/amine compounds, which is
exactly right. Cached to `../../data/raw/`.

## Results (reproducible: `python validate_real.py`)

| component | metric | value |
|---|---|---|
| Goodness model (ECFP4 -> pleasantness, RandomForest) | 5-fold CV Spearman | **0.50** |
| | 5-fold CV R2 | 0.26 |
| Learned mixture model vs mean-pool (non-additive oracle) | Spearman | **0.82 vs 0.25** (+0.57) |
| Safety screen | molecules passing (eligible) | 380 / 420 |
| Suite | all members pass safety screen | yes (hard gate) |
| | mean novelty vs cocoa | ~0.8 |

Spearman 0.50 for structure->pleasantness on ~420 molecules is consistent with the
olfaction literature (pleasantness is among the more predictable attributes). The mixture
result shows a learned set model captures interaction that mean-pooling throws away,
motivating a real learned mixture model over naive additivity.

## The safety / toxicity checker (`safety.py`)

A food-appropriate hazard screen, because drug-discovery filters are miscalibrated for
food: tested directly, Brenk/PAINS flag vanillin (safe) for its aldehyde yet pass benzene
(a carcinogen). Instead this combines (1) a curated denylist of known toxic/restricted
substances, (2) genotoxic structural alerts (nitroaromatic, N-nitroso, azo, aromatic amine,
epoxide, aziridine, hydrazine, diazo), and (3) element/property rules, returning
OK / CAUTION / TOXIC. Only OK molecules enter a suite; CAUTION (e.g. cinnamaldehyde's enal)
and TOXIC are excluded. **This is a screen, not a GRAS determination** - see HANDOFF.md.

## The deliverable: candidate suite

`python suite_real.py` (output in `results/flavor_suite_real.{json,md}`) composes k-molecule
combos from safety-passing real molecules, hill-climbs on mean predicted pleasantness, keeps
only combos chemically novel vs a cocoa key-odorant reference set, and ranks them. Each
candidate lists real molecules + SMILES, predicted pleasantness, novelty, and safety. Combo
score is the mean of member predicted pleasantness (a transparent additive estimate); the
learned mixture model is the documented non-additive refinement (not yet trained on real
mixture labels, so not used to rank).

## Files

- `real_data.py` - real molecules + real pleasantness labels (Keller 2016 via Pyrfume).
- `featurize.py` - RDKit SMILES -> ECFP4.
- `goodness_real.py` - real goodness model + cross-validation.
- `safety.py` - food-appropriate toxicity screen.
- `mixture_model.py` - learned set mixture model vs mean-pool baseline (non-additive oracle).
- `suite_real.py` - the candidate-suite generator.
- `validate_real.py` - pre-registered validation incl. a hard safety gate.
- `test_real_flavor.py` - offline unit tests.
- `results/` - validation JSON + the generated suite (JSON + markdown).
- `HANDOFF.md` - what to give a chemist / molecular gastronomist, and safety caveats.

## Honest limits

Pleasantness R2 is modest (structure->hedonics is genuinely hard and individual-variable).
The combo score assumes additivity; real mixtures are not additive, which is why the learned
mixture model exists and why the top of the ranking should be treated as hypotheses. Nothing
here is a safety clearance. See [HANDOFF.md](HANDOFF.md).

_Conjecture and framing: Jake Wintermute (see repo README). Data: Keller 2016 via Pyrfume
(Castro et al. 2024). Methods, safety screen, and analysis: mine._
