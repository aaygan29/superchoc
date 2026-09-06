# Submission guide

## The paper

- `superchoc.tex` / `superchoc.pdf` - the write-up, **already anonymized** ("Anonymous
  Author(s)", double-blind ready). It uses a plain `article` class; swap in the target venue's
  official style file (e.g. `neurips_2024.sty`) before submitting.

## Honest venue assessment

What we have: a real structure to human-pleasantness benchmark result (5-fold CV Spearman
$0.50$), a de-novo generation + novelty pipeline, a food-appropriate safety screen, a learned
mixture model, compositional statistics, and a machine-checked metric property. What we do
**not** have: wet-lab or human-panel validation of the generated flavors, and pleasantness
$R^2$ is modest. That places the work at **workshop tier**, not main-conference tier, today.

### Recommended targets (via OpenReview)

1. **Best fit - a NeurIPS 2026 workshop** (submit on OpenReview when the workshop CFP opens):
   - *AI for Science* (NeurIPS AI4Science) - method + honest limitations fit well.
   - *Machine Learning for Molecules / ML for Materials* - the de-novo + property-prediction
     framing fits directly.
   - *New in ML* (NeurIPS) - appropriate for a first strong result with a clear roadmap.
2. **Stretch, same family**: **NeurIPS Datasets \& Benchmarks Track** - viable only if reframed
   around the released flavor-design benchmark + artifacts; reviewers will ask for human
   validation, so temper expectations.
3. **Alternatives**: ICLR / ICML *AI4Science* or *ML4Molecules* workshops (OpenReview); or a
   computational food-science / chemical-senses venue.

**Recommendation:** target a NeurIPS 2026 AI-for-Science or ML-for-Molecules **workshop** as
primary, with New in ML as backup. Do a main-track Datasets \& Benchmarks attempt only after
adding at least a small human-panel validation of the top candidates.

### OpenReview mechanics

- Each venue has its own OpenReview submission portal opened from its CFP page; submit the
  anonymized PDF there. Confirm the venue's dual-submission and anonymization policy first.
- Most NeurIPS-family venues are double-blind: keep the PDF anonymous and, if you link code,
  use the anonymized mirror below (not the GitHub URL, which reveals the account).

## Anonymized code mirror (anonymous.4open.science)

The paper can cite an anonymized copy of this repository:

1. Go to https://anonymous.4open.science and create a new anonymized repository from this
   repository's GitHub URL (you must be signed in as the owner).
2. Set an embargo date at/after the decision notification.
3. Paste the generated `anonymous.4open.science/r/...` link into the paper where code is cited.

The repository content itself carries no author name in files, so it is already
content-anonymous; 4open.science additionally hides the GitHub account. Do **not** paste the
raw GitHub URL into a double-blind submission.
