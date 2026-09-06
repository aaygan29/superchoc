# superchoc

A computational research project on **Conjecture 4: Superchocolate exists in flavor
space.**

## Attribution

The conjecture and its framing are **not mine**. They are the work of **Jake Wintermute**,
from his article *"Strange and Marvelous Challenges for Biological AI"* (September 2026),
which proposes a set of Biological Conjectures for Bio-AI teams, published via
[American Wetware](https://americanwetware.com/). Conjecture 4 (Superchocolate) is his,
as are Conjectures 1-3 (Matrigel, the Calvin cycle, bioreactor scaling). All conjecture
text quoted below is his.

This repository is my computational work **toward** his conjecture: a literature library,
a ground-truth dataset corpus, and an online experimental-design method. The ideas being
tested originate with Jake Wintermute; the implementation and analysis here are mine.

## The conjecture, in Wintermute's words

> "We conjecture that the complexity of chocolate implies the possibility of
> superchocolate. [...] there is no particular reason to think that chocolate represents
> a maximum of delight."

The argument, briefly (paraphrased from his article): chocolate has more than
[600 flavor-active volatiles](https://pubmed.ncbi.nlm.nih.gov/33371573/), where vanilla,
banana, and pineapple are each essentially one molecule. Each volatile has a binding
profile across ~30 taste and ~400 odor receptors, so a flavor is a high-dimensional
receptor-activation embedding. Flavor space is vast and mostly unexplored, so exceptional
embeddings no human has tasted likely exist. Because human odor receptors are GPCRs, the
same target class as ~36% of approved drugs
([Nature Rev Drug Discov 2025](https://www.nature.com/articles/s41573-025-01139-y)),
solving generative flavor design plausibly advances generative GPCR drug design.

His three success criteria for Conjecture 4:

1. **Predict** the experience of a flavor from the structure of a molecule.
2. **Generate** a new flavor as unique and delicious as chocolate.
3. **Build** the world-beating model for designing drugs that target GPCRs.

## What is in this repository (my work)

- [ROADMAP.md](ROADMAP.md) - the three success criteria decomposed into concrete mapping problems and milestones.
- [docs/LITERATURE.md](docs/LITERATURE.md) - synthesis of the molecular-flavor and olfaction literature, doubling as a library of experimental techniques and computational tools.
- [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md) - the fixed, public datasets and benchmarks that anchor each criterion, plus the evaluation contract.
- [docs/SYNTHESIS.md](docs/SYNTHESIS.md) - assessment of whether existing work can be bridged, computationally, into progress on Wintermute's three criteria, and a concrete in-silico experiment plan.
- [methods/active_flavor_search/](methods/active_flavor_search/) - a validated online (sequential) experimental-design method for deciding which molecule to taste next, so each human-panel cycle buys maximum progress toward Criterion 2.
- [data/](data/) - dataset scaffold and loaders for the ground-truth corpus (interfaces and sources; does not fabricate data).

## Honesty note

Nothing here is a solved criterion. The method is validated only on a reproducible
synthetic ground truth (does online search beat random screening?), which is a fair test
of the search technique, not a claim about real flavor. Real claims require the public
datasets in [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md) and, for deliciousness, blinded
human panels.
