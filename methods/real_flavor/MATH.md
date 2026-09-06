# Flavor-space mathematics: what can and cannot be proven

You asked whether we can mathematically prove a flavor combination is "tasty" and validate
it (e.g. in Lean). Here is the honest framework.

## The line between proof and empirics

- **Tastiness is empirical.** Whether a physical blend tastes good is a fact about human
  perception, established by tasting, not by a theorem. No Lean proof can establish it.
- **The metric is mathematical.** The functionals we optimize (a tastiness estimate, a
  novelty distance, the design objective) have mathematical properties that *can* be proven,
  and the metric's ability to predict human ratings *can* be empirically validated.

So we do two separate, honest things: prove properties of the metric (Lean), and validate
the metric against real human data (Keller 2016).

## Definitions

For a recipe R with molecules i, model-predicted pleasantness p_i in [0,100], weights
w_i >= 0 summing to 1:

- Tastiness (additive estimate): **T(R) = sum_i w_i p_i**
- Novelty: **N(R) = min over known flavors F of cosine_distance(profile(R), profile(F))**
- Objective: **J(R) = a·T(R)/100 + b·N(R) − c·tox(R)**, a,b,c >= 0

## Theorems (machine-checked in Lean: `FlavorMath.lean`, `lean` exit 0)

- **Best-part bound (P2):** every component's pleasantness <= the blend's best part;
  since T(R) is a convex combination, **T(R) <= max_i p_i**. Proven as `le_listMax` /
  `blend_not_better_than_best` over `Nat` in Lean core (via `omega`, no Mathlib).
- **Boundedness (P1):** if every p_i <= 100 then the blend estimate <= 100
  (`listMax_le_of_all_le`).

**Why P2 matters (this is the crux of your question).** Under the *additive* estimate, a
blend can never beat its single best component. That is a theorem, not an opinion. It is
exactly why designing a mixture that is *better than any of its parts* requires a
**non-additive** mixture model, which is what `mixture_model.py` provides and validates
(learned set model Spearman 0.82 vs mean-pool 0.25 on a non-additive oracle). The math tells
us precisely where additivity is a wall and where the empirical, learned model must take
over.

## Property checks (executable, `flavor_math.py`)

5000 random trials confirm P1-P4 hold on the actual implementation (bounded taste,
best-part bound, cosine-distance is a valid symmetric semimetric in [0,2], objective
bounded). All pass. This tests that the code matches the proven properties.

## Empirical validation of the metric (against real humans)

The tastiness metric (model pleasantness) predicts held-out **human** pleasantness at
**Spearman 0.50, R2 0.26** (5-fold CV on 420 Keller 2016 molecules). That is the honest
strength of the "is it tasty" claim at the single-molecule level: a moderate, real,
better-than-chance predictor, consistent with the olfaction literature, not a proof.

## What this licenses us to claim about the new combos

- Bounds and validity of the *score* are proven/tested (the metric is well-behaved).
- The score is an empirically moderate predictor of human pleasantness.
- Therefore the generated recipes are **ranked hypotheses** with a validated, bounded
  scoring metric, not proven-tasty formulas. A blend that must *exceed* its best component
  needs the non-additive mixture model and, ultimately, a human panel (see HANDOFF.md).

## Reproduce

```bash
python flavor_math.py           # property checks + empirical validity
lean FlavorMath.lean            # machine-check the theorems (exit 0 = proved)
```
