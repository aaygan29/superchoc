# superchoc

A computational research project on **Conjecture 4: Superchocolate exists in flavor
space.**

## Attribution

The conjecture and its framing are **not mine**. They are the work of **Jake Wintermute**,
from his article *"Strange and Marvelous Challenges for Biological AI"* (September 2026),
proposing a set of Biological Conjectures for Bio-AI teams, published via American Wetware
([Substack](https://americanwetware.substack.com/); author
[LinkedIn](https://www.linkedin.com/in/jake-wintermute/)). Conjecture 4 (Superchocolate)
is his, as are Conjectures 1-3 (Matrigel, the Calvin cycle, bioreactor scaling).

This repository is my computational work **toward** his conjecture. The ideas being tested
originate with Jake Wintermute; the implementation and analysis here are mine. Read his
article for the full argument; it is not reproduced here.

## The conjecture, briefly

Wintermute conjectures that because chocolate is such a high-dimensional flavor (hundreds
of odor-active molecules acting across ~30 taste and ~400 odor receptors), and flavor
space is vast and mostly unexplored, there is no reason to think chocolate is the maximum
of delight: other exceptional, unexperienced flavors ("superchocolate") likely exist.
Because odor receptors are GPCRs, the same class as ~36% of approved drugs, solving
generative flavor design may also advance GPCR drug design. See his article for the
argument in full.

His three success criteria for Conjecture 4:

1. **Predict** the experience of a flavor from the structure of a molecule.
2. **Generate** a new flavor as unique and delicious as chocolate.
3. **Build** the world-beating model for designing drugs that target GPCRs.

## What is in this repository (my work)

- [ROADMAP.md](ROADMAP.md) - the three success criteria decomposed into concrete mapping problems and milestones.
- [docs/LITERATURE.md](docs/LITERATURE.md) - synthesis of the molecular-flavor and olfaction literature, doubling as a library of experimental techniques and computational tools.
- [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md) - the fixed, public datasets and benchmarks that anchor each criterion, plus the evaluation contract.
- [docs/SYNTHESIS.md](docs/SYNTHESIS.md) - assessment of whether existing work can be bridged, computationally, into progress on Wintermute's three criteria, and a concrete in-silico experiment plan.
- [docs/CROSS_DISCIPLINE.md](docs/CROSS_DISCIPLINE.md) - how flavor chemistry, receptor biology, neuroscience, and neuropsychology integrate into the pipeline, with the papers that changed specific modeling choices.
- [REPLICATION.md](REPLICATION.md) - every step to reproduce the results, the exact methodology, and how to go from the synthetic demo to a real lab-testable suite.
- [methods/active_flavor_search/](methods/active_flavor_search/) - a validated sequential experimental-design method for deciding which molecule to taste next, so each panel cycle buys maximum progress toward Criterion 2.
- [methods/goodness_model/](methods/goodness_model/) - a validated ML "goodness" model + combo generator on synthetic ground truth, with a Goodhart control so the picks are good on the true oracle, not just in the model's opinion.
- [methods/real_flavor/](methods/real_flavor/) - the **real-data** version: real molecules with real human-panel pleasantness labels (Keller 2016 via Pyrfume), a real goodness model (held-out Spearman 0.50), a learned mixture model that beats mean-pooling on non-additive mixtures, a **food-appropriate toxicity screen**, and a ranked suite of novel, safety-passing candidate flavor combinations with a chef/chemist [handoff](methods/real_flavor/HANDOFF.md).
- [data/](data/) - dataset scaffold and loaders for the ground-truth corpus, plus `online_ingest.py` for live pulls from ChEMBL / PubMed / bioRxiv (does not fabricate data).

## Two senses of "online"

The project uses "online" two ways, both implemented: (1) **live data ingestion** -
continuously pulling from public sources as needed (`data/online_ingest.py`), and folding
newly relevant papers into the methods as concrete design choices (see CROSS_DISCIPLINE.md);
(2) **online/sequential experimental design** - the active-search method that chooses the
next molecule to taste one at a time.

## Honesty note

Nothing here is a solved criterion. The method is validated only on a reproducible
synthetic ground truth (does online search beat random screening?), which is a fair test
of the search technique, not a claim about real flavor. Real claims require the public
datasets in [docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md) and, for deliciousness, blinded
human panels.
