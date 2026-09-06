# FlavorDesigner (unified: chemical + hedonic/taste + odor-profile + toxicity)

**Objective:** maximize 1.0*taste + 0.6*novelty - 0.5*toxicity

> One model over real inputs: pleasantness from a human panel (Keller 2016), odor profile from Leffingwell, chemistry from ECFP4, toxicity from a food-appropriate screen. TOXIC molecules are excluded; CAUTION molecules are penalized so toxicity is actively minimized. Proportions are heuristic starting points. NOT a safety clearance; see HANDOFF.md.

## Proposal 1: "waxy-fatty-brandy accord"  (objective 1.1056)

Taste (pleasantness): 52.78/100 | perceptual novelty 0.963 | chemical novelty 0.69 | toxicity penalty 0.0

Flavor profile: waxy (0.6), fatty (0.6), brandy (0.2), odorless (0.2), buttery (0.2), dairy (0.2), earthy (0.2), camphoreous (0.2)

Batch: 10.0 g = 1.0 g aroma in 9.0 g propylene glycol (food grade).

| molecule | % aroma | mg/batch | odor notes | CAS | safety |
|---|---|---|---|---|---|
| Isoeugenol | 39.79 | 397.9 | n/a | 97-54-1 | OK |
| isoeugenol acetate | 23.09 | 230.9 | n/a | 93-29-8 | OK |
| Isosafrole | 9.09 | 90.9 | n/a | 120-58-1 | OK |
| Ethyl stearate | 4.53 | 45.3 | waxy, odorless | 111-61-5 | OK |
| Isopropyl myristate | 4.33 | 43.3 | fatty, oily | 110-27-0 | OK |
| (-)-beta-Pinene | 4.02 | 40.2 | n/a | 18172-67-3 | OK |
| Camphor | 3.55 | 35.5 | mint, camphoreous, fresh | 76-22-2 | OK |
| 6-Undecanone | 2.96 | 29.6 | fatty, waxy, cheesy | 927-49-1 | OK |
| Water | 2.65 | 26.5 | n/a | 7732-18-5 | OK |
| D-(-)-Ribose | 2.62 | 26.2 | n/a | 50-69-1 | OK |
| Butyl 10-undecenoate | 2.04 | 20.4 | winey, brandy, buttery | 109-42-2 | OK |
| L-Histidine | 1.31 | 13.1 | n/a | 71-00-1 | OK |

## Proposal 2: "fresh-mint-camphoreous accord"  (objective 1.1047)

Taste (pleasantness): 54.91/100 | perceptual novelty 0.926 | chemical novelty 0.802 | toxicity penalty 0.0

Flavor profile: fresh (0.6), mint (0.4), camphoreous (0.4), ketonic (0.2), ethereal (0.2), solvent (0.2), leafy (0.2), green (0.2)

Batch: 10.0 g = 1.0 g aroma in 9.0 g propylene glycol (food grade).

| molecule | % aroma | mg/batch | odor notes | CAS | safety |
|---|---|---|---|---|---|
| Isoeugenol | 35.91 | 359.1 | n/a | 97-54-1 | OK |
| 6-Acetyl-1,1,2,4,4,7-Hexamethyltetralin | 14.95 | 149.5 | n/a | 21145-77-7 | OK |
| Methyl 2-nonynoate | 10.29 | 102.9 | green, violet, leafy | 111-80-8 | OK |
| Isosafrole | 8.21 | 82.1 | n/a | 120-58-1 | OK |
| Ambrox | 8.16 | 81.6 | n/a | 6790-58-5 | OK |
| 6-Methylcoumarin | 4.61 | 46.1 | coconut, coumarinic | 92-48-8 | OK |
| Butylated hydroxytoluene | 3.97 | 39.7 | n/a | 128-37-0 | OK |
| D-Camphor | 3.21 | 32.1 | n/a | 464-49-3 | OK |
| Camphor | 3.21 | 32.1 | mint, camphoreous, fresh | 76-22-2 | OK |
| 1,8-cineole | 3.19 | 31.9 | camphoreous, fresh | 470-82-6 | OK |
| Butylated hydroxyanisole | 2.82 | 28.2 | n/a | 25013-16-5 | OK |
| 3-Methylcyclohexanone | 1.47 | 14.7 | solvent, ketonic, fresh | 591-24-2 | OK |

## Proposal 3: "herbal-winey-fatty accord"  (objective 1.0867)

Taste (pleasantness): 56.35/100 | perceptual novelty 0.872 | chemical novelty 0.689 | toxicity penalty 0.0

Flavor profile: herbal (0.38), winey (0.38), fatty (0.25), oily (0.25)

Batch: 10.0 g = 1.0 g aroma in 9.0 g propylene glycol (food grade).

| molecule | % aroma | mg/batch | odor notes | CAS | safety |
|---|---|---|---|---|---|
| vanillin | 57.35 | 573.5 | vanilla, creamy | 121-33-5 | OK |
| Isoeugenol | 16.5 | 165.0 | n/a | 97-54-1 | OK |
| isoeugenol acetate | 9.58 | 95.8 | n/a | 93-29-8 | OK |
| Isosafrole | 3.77 | 37.7 | n/a | 120-58-1 | OK |
| 7-Methoxycoumarin | 3.26 | 32.6 | n/a | 531-59-9 | OK |
| Isoamyl laurate | 1.89 | 18.9 | winey, oily, fatty | 6309-51-9 | OK |
| 6-Methyl-5-hepten-2-one | 1.72 | 17.2 | herbal, green, oily | 110-93-0 | OK |
| Camphor | 1.47 | 14.7 | mint, camphoreous, fresh | 76-22-2 | OK |
| DL-Tartaric acid | 1.41 | 14.1 | odorless | 133-37-9 | OK |
| Methyl 2-methoxybenzoate | 1.4 | 14.0 | herbal, winey, anisic | 606-45-1 | OK |
| Butyl 10-undecenoate | 0.85 | 8.5 | winey, brandy, buttery | 109-42-2 | OK |
| 1-Methyl-3-methoxy-4-isopropylbenzene | 0.79 | 7.9 | spicy, herbal | 1076-56-8 | OK |

## Proposal 4: "fatty-winey-mint accord"  (objective 1.0803)

Taste (pleasantness): 53.25/100 | perceptual novelty 0.913 | chemical novelty 0.684 | toxicity penalty 0.0

Flavor profile: fatty (0.29), winey (0.29), mint (0.29)

Batch: 10.0 g = 1.0 g aroma in 9.0 g propylene glycol (food grade).

| molecule | % aroma | mg/batch | odor notes | CAS | safety |
|---|---|---|---|---|---|
| Isoeugenol | 43.35 | 433.5 | n/a | 97-54-1 | OK |
| Isosafrole | 9.91 | 99.1 | n/a | 120-58-1 | OK |
| diethyl succinate | 9.08 | 90.8 | winey, green, grape | 123-25-1 | OK |
| benzaldehyde | 8.5 | 85.0 | almond | 100-52-7 | OK |
| delta-Undecalactone | 5.18 | 51.8 | fatty, peach, creamy | 710-04-3 | OK |
| Isoamyl laurate | 4.96 | 49.6 | winey, oily, fatty | 6309-51-9 | OK |
| butanediol | 3.88 | 38.8 | n/a | 513-85-9 | OK |
| D-Camphor | 3.87 | 38.7 | n/a | 464-49-3 | OK |
| D-Dihydrocarvone | 3.74 | 37.4 | mint | 7764-50-3 | OK |
| Water | 2.89 | 28.9 | n/a | 7732-18-5 | OK |
| 2-sec-Butylcyclohexanone | 2.56 | 25.6 | mint | 14765-30-1 | OK |
| 1-Methyl-3-methoxy-4-isopropylbenzene | 2.08 | 20.8 | spicy, herbal | 1076-56-8 | OK |

## Proposal 5: "ethereal-mint-creamy accord"  (objective 1.0582)

Taste (pleasantness): 55.96/100 | perceptual novelty 0.831 | chemical novelty 0.692 | toxicity penalty 0.0

Flavor profile: ethereal (0.44), mint (0.22), creamy (0.22), rum (0.22), ketonic (0.22)

Batch: 10.0 g = 1.0 g aroma in 9.0 g propylene glycol (food grade).

| molecule | % aroma | mg/batch | odor notes | CAS | safety |
|---|---|---|---|---|---|
| vanillin | 56.03 | 560.3 | vanilla, creamy | 121-33-5 | OK |
| Isoeugenol | 16.12 | 161.2 | n/a | 97-54-1 | OK |
| isoeugenol acetate | 9.36 | 93.6 | n/a | 93-29-8 | OK |
| eugenol | 6.28 | 62.8 | pungent, dry, smoky | 97-53-0 | OK |
| 7-Methoxycoumarin | 3.19 | 31.9 | n/a | 531-59-9 | OK |
| tert-butanol | 2.84 | 28.4 | ethereal, musty, fermented | 75-65-0 | OK |
| 2-decanone | 1.99 | 19.9 | orange, floral | 693-54-9 | OK |
| 6-Undecanone | 1.2 | 12.0 | fatty, waxy, cheesy | 927-49-1 | OK |
| Ethyl formate | 1.01 | 10.1 | winey, ethereal, cognac | 109-94-4 | OK |
| Butyl formate | 0.88 | 8.8 | plum, rum, ethereal | 592-84-7 | OK |
| 3-Methylcyclohexanone | 0.66 | 6.6 | solvent, ketonic, fresh | 591-24-2 | OK |
| Cycloheptanone | 0.45 | 4.5 | mint, ketonic | 502-42-1 | OK |

## How to make
Weigh each component (mg) into an amber vial (fume hood), add food-grade propylene glycol to the carrier mass, mix, equilibrate 24-48 h, evaluate by smell first. Do NOT taste without GRAS/FEMA confirmation (HANDOFF.md).