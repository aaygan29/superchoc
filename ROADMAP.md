# Roadmap

Three success criteria, each broken into the mapping problems it actually requires. The
through-line: molecule structure to receptor activation to perceived flavor, learned
against human report, with GPCR biology shared across flavor and drug design.

## Criterion 1: Predict the experience of a flavor from molecular structure

The forward model. Given a molecule (or a mixture), predict what a human tastes.

- **Structure to receptor activation.** Predict binding and activation across the ~30
  taste and ~400 odor receptors from molecular structure. Odor receptors are GPCRs, so
  this shares machinery with Criterion 3.
- **Receptor activation to percept.** Map the receptor activation profile to human
  flavor descriptors. This is the layer that must be learned from human tasters, not
  from chemistry alone.
- **Mixtures, not just single molecules.** Real flavors are dozens to hundreds of
  compounds. Model suppression, synergy, and masking rather than assuming additivity.
- **Data.** Assemble a structure/receptor/descriptor corpus. Candidate anchors: the
  Principal Odor Map line of work, GoodScents / Leffingwell odorant sets, and any
  panel-tasting data that pairs mixtures with human descriptors.
- **Milestone.** Held-out molecules and mixtures whose predicted descriptor profile
  matches blinded human panels above a stated baseline.

## Criterion 2: Generate a new flavor as unique and delicious as chocolate

The inverse model. Search flavor space for embeddings that no human has tasted and that
a panel rates as exceptional.

- **Define the objective.** A "deliciousness" signal grounded in human report, not a
  proxy the optimizer can game. Guard against Goodharting the forward model.
- **Generative search.** Propose novel molecule mixtures conditioned on a target region
  of flavor space, using Criterion 1 as the in-silico evaluator.
- **Safety and feasibility filter.** Screen candidates for edibility, toxicity, and
  synthesizability before anything reaches a human.
- **Human-in-the-loop design cycles.** Cheap and safe relative to pharma: synthesize or
  blend top candidates, run taste panels, feed results back. Iterate.
- **Milestone.** At least one novel embedding a blinded panel rates comparable to or
  above high-quality chocolate.

## Criterion 3: World-beating model for designing drugs that target GPCRs

The transfer bet. Odor receptors are GPCRs, so the structure-to-GPCR-activation model
built for flavor is the same model class used for drug design.

- **GPCR pocket and binding prediction.** Structural pocket prediction and
  ligand/activation modeling across the GPCR family (Liu et al., 2024 line of work).
- **Generative ligand design.** Propose small molecules for a chosen GPCR target with
  desired agonist/antagonist profile.
- **ADMET and selectivity.** Off-target and pharmacokinetic modeling, benchmarked
  against public medicinal-chemistry data (e.g. ChEMBL).
- **Benchmark honestly.** Compare against published GPCR docking and affinity baselines
  on held-out targets; report where it wins and where it does not.
- **Milestone.** Beat a strong published baseline on a held-out GPCR affinity or
  activity benchmark.

## Shared infrastructure

- Molecular representation and featurization used by all three criteria.
- A single GPCR activation model reused between odor (C1) and drug design (C3).
- A human-report data pipeline: panel protocols, descriptor ontology, storage schema.
- Evaluation harness with pre-registered baselines and blinded panels, so "as delicious
  as chocolate" and "world-beating" are measured, not asserted.

## Progress against this roadmap (current build)

**Criterion 1 (forward model).**
- Structure -> percept: pleasantness predicted from structure (Keller, CV Spearman 0.50) and,
  strikingly, from flavor-space *position* alone (goodness field, 0.49). `goodness_real.py`,
  `flavor_field.py`.
- Structure -> receptor: real molecule<->OR signal at 1.57x chance (Mainland NN), plus a
  structure-based Boltz-2 co-fold that separates cognate from non-cognate ligands (AUC 0.875).
  `receptors.py`, `receptor_binding.py`. (A full molecule->OR model, e.g. M2OR, is the next step.)
- Receptor/percept -> descriptors: Leffingwell odor descriptors + five basic tastes + geometric
  flavor regions. `flavor_profile.py`, `taste_profile.py`.
- Mixtures, not additivity: mixture perceptual-distance metric validated on real human ratings
  (Snitz -0.49, Ravia -0.245, both FDR-significant); learned non-additive vs mean-pool on a
  synthetic oracle (0.82 vs 0.25); omission testing and an olfactory-white guardrail. `mixtures.py`.

**Criterion 2 (inverse model).**
- Objective: a recipe-level good-flavor classifier over 23 pleasant, chemotype-diverse flavors,
  benchmarked leave-one-flavor-out (AUC 0.72, RF 0.73). Note: adding savory/pungent classes to
  force chemotype balance HURT it (0.58), so the flavor set stays inside genuinely pleasant
  flavors; ester/lactone prevalence reflects real pleasantness lifts, not bias. `recipe_classifier.py`.
- Generative search: de-novo molecules (`lego_assembly.py`) gated for 3D validity
  (`geometry_gate.py`) and safety (`safety.py`), assembled into ranked recipes with optimized
  ratios by the overarching `analyzer.py`; sparse-yet-good regions of the hyperbolic space are
  the superchocolate targets (`flavor_field.superchoc_score`).
- Safety/feasibility filter: in place. Human-in-the-loop panels: not yet (the honest gap).

**Criterion 3 (GPCR transfer).** Only the shared receptor-binding machinery (Boltz-2) exists;
therapeutic-GPCR benchmarking is future work.

## Open questions (updated with what we have learned)

- How additive is flavor space? Ratio optimization collapses to a single component under any
  additive objective (the Lean best-part bound); genuine blends require a validated non-additive
  response, which we approximate with the classifier but have not validated on human combo data.
- What is the right deliciousness objective? The current classifier learns "looks like a real
  flavor" (known flavors vs random), a coherence prior, NOT measured deliciousness. Cross-chemotype
  generalization (AUC 0.58) shows it partly keys on chemotype familiarity. Real human
  combo-pleasantness data is the missing ground truth.
- Chemical novelty is not perceptual novelty: de-novo-heavy recipes are chemically novel but
  perceptually familiar and score lower on good-flavor. The tool measures this tension explicitly.
- How much does the flavor GPCR model transfer to therapeutic targets? Untested.
