"""Neuro-reward layer: food molecules that engage reward / mood / arousal pathways.

Flavor pleasantness is not the only reason a food feels good. Some food-borne molecules act
on neuromodulatory systems: caffeine (adenosine antagonism -> alertness), theobromine (mild
stimulant, in cocoa), phenylethylamine (trace-amine / dopaminergic, in cocoa), tryptophan
(serotonin precursor), tyrosine (dopamine precursor). This is the "neuro" axis: a designed
flavor can optionally carry a documented functional/mood dimension, not just taste.

This is a CURATED, cited reference set with mechanism and typical dietary source. It is used
only to (a) annotate and (b) optionally add a small, dose-limited functional component to a
blend. It makes NO health claim: these are psychoactive/active at dose, so any inclusion is
subject to strict dose limits and regulatory review (see HANDOFF.md). Nothing here is a
recommendation to ingest.
"""

from __future__ import annotations

# name -> mechanism, target/pathway, typical food source, canonical SMILES, caution.
NEURO_REWARD = {
    "caffeine": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "pathway": "adenosine A1/A2A antagonist -> alertness/arousal",
        "source": "coffee, tea, cocoa", "caution": "stimulant; strict dose limit"},
    "theobromine": {
        "smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "pathway": "adenosine antagonist (milder than caffeine); mood/mild stimulant",
        "source": "cocoa", "caution": "stimulant; toxic to some animals (dogs)"},
    "theophylline": {
        "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "pathway": "adenosine antagonist / PDE inhibitor -> arousal",
        "source": "tea, cocoa", "caution": "narrow therapeutic index; strict limit"},
    "phenylethylamine": {
        "smiles": "NCCc1ccccc1",
        "pathway": "trace amine / dopaminergic-noradrenergic release ('chocolate amine')",
        "source": "cocoa", "caution": "rapidly metabolized (MAO); amine"},
    "L-tryptophan": {
        "smiles": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
        "pathway": "serotonin (5-HT) precursor -> mood/satiety",
        "source": "many proteins; turkey, cocoa", "caution": "amino acid; dietary amounts"},
    "L-tyrosine": {
        "smiles": "NC(Cc1ccc(O)cc1)C(=O)O",
        "pathway": "dopamine precursor -> motivation/arousal",
        "source": "cheese, cocoa", "caution": "amino acid; dietary amounts"},
    "vanillin": {  # also a flavor; mild documented mood association
        "smiles": "O=Cc1ccc(O)c(OC)c1",
        "pathway": "olfactory-hedonic; associated calming effect",
        "source": "vanilla, cocoa", "caution": "generally recognized safe as flavor"},
}


def as_candidates():
    """Return the neuro-reward molecules as pool-style rows for the designer to consider."""
    return [{"name": k, "smiles": v["smiles"], "pathway": v["pathway"],
             "source": v["source"], "caution": v["caution"]} for k, v in NEURO_REWARD.items()]


if __name__ == "__main__":
    from rdkit import Chem
    for k, v in NEURO_REWARD.items():
        ok = Chem.MolFromSmiles(v["smiles"]) is not None
        print(f"{k:18s} valid={ok}  {v['pathway']}")
