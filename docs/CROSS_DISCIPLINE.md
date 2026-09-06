# Cross-disciplinary integration

Wintermute's Conjecture 4 sits at the intersection of several fields. Progress needs their
methods wired together, not treated separately. This maps how each discipline's results
feed the pipeline, and how the repo pulls from them continuously ("online"). Extends
[LITERATURE.md](LITERATURE.md); biomedical citations retrieved via PubMed (DOIs below).

## The four layers and what each contributes

| Discipline | Question it answers | Method we borrow | Where it enters the pipeline |
|---|---|---|---|
| **Flavor chemistry** | Which molecules are in chocolate, and which are smelled? | GC-olfactometry, odor activity value, aroma recombination | Defines the target (chocolate key odorants) and the molecule library |
| **Receptor biology** | Which receptor does a molecule activate? | GPCR/OR structure prediction, docking, ML activation models | Molecule -> receptor-activation embedding (bridge to drug design) |
| **Neuroscience** | How are mixture signals combined? | Single-neuron electrophysiology of odor mixtures | Justifies (and will replace) the mixture aggregator |
| **Neuropsychology** | What makes a smell pleasant? | Perceptual-dimension analysis, hedonic ratings | Defines "goodness" as a learnable scalar target |

## Key integrations and their evidence

- **Mixtures are non-additive (neuroscience).** Single olfactory sensory neurons integrate
  the components of a mixture with suppression, hypoadditivity, and synergy; a mixture's
  response is not predictable from its components, and a minor component can dominate the
  percept. This is exactly why the goodness generator's mean-pooling aggregator is a
  first-order stand-in that must become a learned mixture model on real data.
  Duchamp-Viret et al. 2003, *Eur J Neurosci*.
  [DOI](https://doi.org/10.1111/j.1460-9568.2003.03001.x)
- **Pleasantness is a primary perceptual axis (neuropsychology).** Odor perceptual space
  is relatively low-dimensional and many studies converge on hedonic valence
  ("pleasantness") as one of its most important dimensions. This licenses modeling
  "goodness" as a single learnable scalar, the target of the goodness model.
  Barnum & Hong 2022, *Curr Biol* "Olfactory coding".
  [DOI](https://doi.org/10.1016/j.cub.2022.10.067)
- **Structure -> percept is learnable and biology-tracking (chemistry + ML).** The
  Principal Odor Map reaches human-level odor prediction and its coordinates track receptor
  and neural responses (see LITERATURE.md: Lee et al. 2023; Qian et al. 2023). This is the
  embedding the goodness model consumes.
- **OR = GPCR (receptor biology).** Odor receptors are GPCRs, so the same
  structure-to-activation machinery serves flavor and drug design, and GPCRdb 2025 now
  hosts both (see GROUND_TRUTH.md).

## "Online": continuous ingestion + literature integration

Two senses, both implemented:

1. **Live data ingestion.** `data/online_ingest.py` pulls on demand from public REST APIs
   (ChEMBL targets/bioactivities, PubMed via NCBI E-utilities, bioRxiv by date), caching to
   `data/raw`. So the corpus refreshes continuously rather than shipping frozen. Verified
   against the live ChEMBL and PubMed endpoints.
2. **Methodology integration.** New relevant papers (like the two above) are folded into the
   method as concrete design decisions, not just a reading list: each citation here changed
   a modeling choice (mixture aggregator, goodness-as-scalar, embedding source).

## Open cross-disciplinary problems

- A learned mixture model that captures suppression/synergy, replacing mean-pooling.
- A goodness label at scale: hedonic ratings are sparse and individual-variable, so
  per-person modeling (as in the DREAM work) is likely required.
- Closing the loop to the therapeutic-GPCR side: does a mixture/OR model transfer to
  drug-target GPCR affinity prediction?

_Conjecture: Jake Wintermute. Neuroscience/neuropsychology citations via PubMed._
