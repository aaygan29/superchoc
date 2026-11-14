# Sources and access log

Rule followed: every numeric value was typed from a table/text I actually opened (PDF text extracted locally, or a fetched page). PubChem (REST, fetched in this task) is the source for SMILES, CAS (first CAS-format synonym) and FEMA numbers (parsed from PubChem synonyms; left blank where PubChem listed conflicting FEMA numbers, e.g. phenylacetaldehyde). Odor-threshold media are recorded as the source states them.

## Sources actually used for numbers (what was accessible)
| key | access | used for |
|---|---|---|
| schlueter2022 | full text (author accepted manuscript, ZHAW repository) | FD (Table 1), concentrations (Table 2), thresholds + OAVs (Table 3) for 3 dark chocolates; thresholds stated as orthonasal in oil |
| ullrich2022 | full text (AAM, ZHAW repository) | FD (Table 2), concentrations (Table 3), thresholds + dose-over-threshold (Table 4), 6 dark chocolates. Table 4 footnote cites reference numbers per threshold but does NOT state the medium; recorded as "not stated" |
| schlueter2020 | full text of published article as reprinted inside the 2023 Hamburg dissertation (PDF) | concentrations/thresholds/OAVs for 3 cocoa nib materials |
| schlueter2023_diss_secondary | full text of dissertation (German) | Table 1 OAVs of Frauendorfer 2008/2019 (SECONDARY quotation, OAV only, no concentrations/thresholds); statements on recombination/omission status |
| tuenter2020 | full text (AAM, Antwerp repository) | Table 3 OAVs; HS-SPME semi-quantitative, thresholds from van Gemert 2011 (oil); no absolute concentration entered |
| cordero2019 | full text (OA manuscript, Univ. Torino) | Table 2 thresholds (ng/g, oily matrix, compiled from literature) for GRAS file only |
| belitz2009 | 14 KB excerpt of ch. 5 (Tables 5.2, 5.3) hosted at wekivaculinary.org | water thresholds for GRAS file |
| frauendorfer2006, frauendorfer2008, frauendorfer2019, seyfried2019, chetschik2019, schnermann1997, counet2002, liu2015 | abstracts only (Europe PMC API / TUM portal) | recombination file (2006, 2019 Seyfried, Liu) and bibliographic info. Full tables NOT accessed (ACS paywall, 403) |
| aprotosoaie2016 | full-text PDF downloaded, text extracted; contains no OAV/concentration table (searched for OAV/threshold) | nothing |
| czerny2008 | abstract/bibliographic record only | nothing (thresholds are in air; not accessed) |

## Looked for but NOT obtained (left blank)
- Per-compound concentrations, thresholds, OAVs and FD factors from Frauendorfer & Schieberle 2006 (cocoa powder), 2008 (Criollo) and 2019 (Forastero) primary tables; only secondary OAVs for 2008/2019 via the dissertation.
- Seyfried & Granvogl 2019 tables (90% / 99% chocolate) and its recombinate panel size/scores.
- Schnermann & Schieberle 1997, Counet 2002, Liu 2015, Chetschik 2019 tables (milk chocolate, Chinese chocolates): abstracts only; Liu 2015 addition-test details unavailable.
- Rottiers 2019, Kongor 2016 (not found open access); Aprotosoaie 2016 has no OAV table.
- Leffingwell odorthre page: HTTPS certificate could not be verified from this environment and I did not bypass it; archive copy blocked. No Leffingwell values entered.
- Czerny 2008 thresholds; thresholds for ethyl vanillin, ethyl maltol, delta-decalactone, beta-ionone, sotolon in water, 2-acetyl-1-pyrroline, 2-acetylthiazole, methional (only sotolon-as-3-hydroxy-4,5-dimethylfuran-2(5H)-one in oil via Ullrich 2022 was found). These GRAS compounds are in PubChem lookups but have no threshold rows.
- Any omission-test results: none found; dissertation states none had been done for cocoa/chocolate as of 2023.

## Doubts and flags
- Thresholds differ greatly between sources for the same compound (e.g. dimethyl trisulfide 0.03 ug/kg in Schlueter 2022 / Ullrich 2022 vs 2.5 in Schlueter 2020; 2-methoxyphenol 1.8 vs 16; ethyl 2-methylbutanoate 0.37 vs 0.26). Rows are kept side by side; do not merge blindly.
- Water thresholds from Belitz are given per litre (mg/l, ug/l) and are entered as ug/kg assuming density 1.
- Ullrich 2022 "oav" column is a dose-over-threshold (DoT) factor, the same ratio as OAV.
- Schlueter 2020: 2-phenylethanol OAV cells garbled in text extraction (left blank); ethyl 3-methylbutanoate unfermented concentration ambiguous (left blank).
- FD factors for "2- and 3-methylbutanal / -butanoic acid" are joint values and are repeated on both isomers.
- "Matrix" for Schlueter/Ullrich is the finished chocolate (not cocoa powder); target-study matrices (cocoa powder, milk chocolate) have no numeric rows.
- CAS = first CAS-format synonym from PubChem; racemic/unspecified stereo SMILES where PubChem name lookup was unspecified. 2-Ethyl-3,6-dimethylpyrazine resolves to PubChem's 3-ethyl-2,5-dimethylpyrazine (same compound).
