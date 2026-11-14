# Candidate molecule pool (prospective binary-mixture tasting study)
Built by `methods/chocolate_anchor/candidate_pool.py` -> `pool.csv` (78 rows); evidence and URLs checked
are in `evidence.json` (2026-11-13). Tests: `test_candidate_pool.py` (offline).

## Counts
| role | rows | eligible | ineligible |
|---|---|---|---|
| cocoa_key (OAV>=1 in >=1 cocoa/chocolate source in cocoa_key_odorants.csv) | 31 | 18 | 13 |
| partner | 47 | 30 | 17 |
| total | 78 | 48 | 30 |
Pool sources: 35 cocoa-file odorants + 29 GRAS-threshold-file molecules (22 overlap = 42 distinct), plus 36
added confectionery odorants (25 eligible). All matrices in the cocoa file are cocoa/chocolate/nib/liquor
materials, so any OAV>=1 counts. `oav_max` and `oav_sources` give the maximum and the source keys.

## Exclusion reasons (a row can have several)
- In Ma 2021 (test data, kept in CSV, eligible=False): 26 rows (13 cocoa_key, 13 partner).
- FDA 2018 delisting: 3 (myrcene, pyridine, pulegone; myrcene is also in Ma 2021).
- Hazard screen TOXIC: 1 (methyl anthranilate, aromatic-primary-amine alert; FEMA/EU listed but screened out).
- `in_keller` is a flag only (39 rows; 19 eligible). Keller 2016 is not a test set here.

## How regulatory status was checked (every URL is in `regulatory_evidence`)
1. FEMA GRAS: search `https://www.femaflavor.org/flavor-library/search?fulltext=<CAS>` (name as fallback), open
   the library page, accept only if the page's own CAS list contains our CAS; FEMA number read from the page.
2. EU Union List: Reg. (EU) 872/2012 annex (adopting the Union List of Reg. 1334/2008 Annex I Part A),
   fetched from `http://publications.europa.eu/resource/celex/32012R0872` (eur-lex.europa.eu itself
   serves a bot challenge). Matched by CAS; FL number and "Restrictions of Use" column recorded.
   This is the text as adopted in 2012, NOT the current consolidated list (not retrievable here), so
   later EU amendments/delistings are not applied. 47 of 48 eligible rows also have an EU FL number.
3. FDA delisting: seven substances parsed from Federal Register 83 FR 50490 (2018-10-09, Food Additive
   Regulations; Synthetic Flavoring Agents and Adjuvants): https://www.federalregister.gov/documents/2018/10/09/2018-21807/food-additive-regulations-synthetic-flavoring-agents-and-adjuvants
   CAS set: benzophenone, ethyl acrylate, eugenyl methyl ether, myrcene, pulegone, pyridine, styrene
   (styrene: removed in a companion rule as abandoned). Only myrcene, pulegone, pyridine are in the pool,
   kept on purpose to exercise the exclusion (FEMA numbers still exist for them).
4. PubChem FEMA annotation fetched per CID as a cross-check (no disagreements found).
5. Hazard screen: `methods/real_flavor/safety.screen` output stored verbatim as JSON. CAUTION (enal etc.,
   9 eligible rows) is not an exclusion; TOXIC/INVALID is.

## Unverified / weak
- beta-damascenone (CAS 23726-93-4): INELIGIBLE ("regulatory evidence only via parent CAS / PubChem annotation").
  The FEMA library entry (FEMA 3420) lists only the parent CAS 23696-85-7 and PubChem's annotation on the exact
  CID also says 3420, but there is no FEMA page for this CAS and no EU row for it, so the pool's own rule excludes it.
- 2-acetyl-1-pyrroline (FEMA 4249): FEMA only, no EU FL in the 2012 annex; it is unstable and hard to
  purchase as a clean standard, so check supplier availability before use.
- Purchasability and EU Annex III maximum levels were not checked (only the Union List restriction column).

## Caveat
FEMA GRAS and EU Union List status cover use as a flavouring ingredient in food at customary levels. They
do not show that a panel stimulus is safe at any concentration or by orthonasal presentation at neat or
high strength. The protocol must use customary dilutions and have ethics approval; the hazard screen is a
filter, not a safety determination.
