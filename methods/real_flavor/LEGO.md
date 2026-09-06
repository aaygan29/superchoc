# De-novo "lego" molecule assembly

Builds NEW molecules from fragments of known safe flavor compounds, then predicts their
taste/odor/receptor engagement and checks whether they are structurally novel.

## Pipeline (`lego_assembly.py`)

1. **Fragment** safe flavor molecules with BRICS (retrosynthetically sensible bonds).
2. **Reassemble** with `BRICS.BRICSBuild` into new molecules; sanitize; dedupe by InChIKey;
   drop anything already in the training set; keep food-plausible size (MW 60-350).
3. **Predict** for each: food-safety verdict; pleasantness (goodness model, Keller 2016);
   odor notes and odorant-receptor hits transferred from the nearest known molecule, with
   the similarity reported so off-distribution guesses can be discounted.
4. **Empirical structure novelty**: query PubChem by InChIKey. No hit = not in PubChem
   (novel); hit = known.

## Result (`results/lego_denovo.json`, seed 0)

- 134 fragments -> 437 assembled -> **434 safe and novel-vs-training**.
- Of the top-12 by predicted pleasantness, **12/12 are absent from PubChem** (structurally
  novel). Example top candidate: `CCCCCCC(=O)Oc1cc(C=O)ccc1O` (a vanillin-heptanoate ester
  hybrid), predicted pleasantness 76.6, notes vanilla/sweet, nearest-known Tanimoto 0.55.

## Honest limits

- Predictions for de-novo molecules are **extrapolation**: the goodness model was trained on
  known odorants, so confidence drops as nearest-known similarity drops (reported per
  molecule). Treat low-similarity predictions with caution.
- Absence from PubChem is strong evidence of novelty, not proof of never-synthesized.
- A novel molecule is **not** cleared for use: it has no GRAS status and no safety data. The
  structural screen is a filter, not a clearance. A new molecule would need real tox testing
  before any human exposure (see HANDOFF.md). These are computational hypotheses for a chemist.

## Reproduce

```bash
python lego_assembly.py   # fragments -> assembles -> predicts -> PubChem novelty check
```
