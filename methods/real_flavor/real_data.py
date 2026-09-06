"""Real molecules with real labels, pulled through the proper channel (Pyrfume).

Source: Keller & Vosshall 2016, "Olfactory perception of chemically diverse molecules"
(the DREAM-challenge dataset), distributed by Pyrfume. ~480 monomolecular odorants, each
rated by a panel on "HOW PLEASANT IS THE SMELL?" (0-100). We take the high-concentration
stimulus (1/1000) and average pleasantness across subjects to get one real pleasantness
label per molecule, joined to its canonical SMILES.

This replaces the synthetic goodness label with a real, human-panel label. The processed
table is cached to data/raw so downstream steps do not re-download.

Refs: Keller et al. 2016 (Pyrfume: Castro et al. 2024, Nature Sci Data
https://www.nature.com/articles/s41597-024-04051-z).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def build_keller_pleasantness(concentration_ratio="1/1000", cache_path=None) -> pd.DataFrame:
    import pyrfume

    mols = pyrfume.load_data("keller_2016/molecules.csv").reset_index()
    beh = pyrfume.load_data("keller_2016/behavior.csv")
    stim = pyrfume.load_data("keller_2016/stimuli.csv").reset_index()

    # pleasantness rows, numeric
    mv = beh["MeasurementValue"].astype(str).str.upper()
    pl = beh[mv.str.contains("PLEASANT", na=False)].copy()
    pl["num"] = pd.to_numeric(pl["Value"], errors="coerce")
    pl = pl.dropna(subset=["num"])

    # mean pleasantness per stimulus
    stim_pl = pl.groupby("Stimulus")["num"].mean().rename("pleasantness").reset_index()

    # keep single-molecule stimuli at the chosen concentration
    s = stim.copy()
    s["CIDs"] = s["CIDs"].astype(str)
    single = s[~s["CIDs"].str.contains(r"[;, ]", regex=True, na=False)]
    single = single[single["Ratio"].astype(str) == concentration_ratio]
    single["CID"] = pd.to_numeric(single["CIDs"], errors="coerce")

    df = single.merge(stim_pl, on="Stimulus", how="inner")
    df = df.merge(mols[["CID", "OdorName", "CanonicalSMILES"]], on="CID", how="inner")
    df = df.rename(columns={"OdorName": "name", "CanonicalSMILES": "smiles"})
    df = df[["CID", "name", "smiles", "pleasantness"]].dropna().drop_duplicates("CID")
    df = df.reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    df = build_keller_pleasantness()
    print("molecules:", len(df))
    print("pleasantness: mean=%.1f std=%.1f min=%.1f max=%.1f"
          % (df.pleasantness.mean(), df.pleasantness.std(),
             df.pleasantness.min(), df.pleasantness.max()))
    print(df.sort_values("pleasantness", ascending=False).head(6)[["name", "pleasantness"]].to_string())
    print(df.sort_values("pleasantness").head(4)[["name", "pleasantness"]].to_string())
