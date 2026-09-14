"""Reference flavors as their published key-odorant sets (SMILES).

Used to (a) anchor "novelty of the combination": a candidate recipe should be distinct
from every known flavor here, not just from cocoa; and (b) sanity-check that a candidate
is not simply a rediscovery of vanilla/coffee/etc.

These are representative key aroma compounds from the flavor-chemistry literature (identified by
aroma extract dilution analysis / OAV / aroma-recombination studies). They are NOT exhaustive
formulas; they are enough to place a candidate relative to known flavors. Per-flavor attribution
to the canonical key-odorant literature:
  cocoa/chocolate  - Frauendorfer & Schieberle (2006), key odorants of cocoa.
  coffee           - Czerny, Mayer & Grosch (1999), potent odorants of roasted coffee.
  butter           - Schieberle, Gassenmeier, Guth et al.; Widder & Schieberle, butter aroma.
  apple            - Schieberle et al., key odorants of apple (ethyl 2-methylbutanoate et al.).
  orange           - Buttery et al., orange essence/juice volatiles.
  peach            - lactone-dominated key-odorant literature (gamma/delta-decalactone).
  honey            - phenylacetaldehyde / 2-phenylethanol honey-aroma studies.
  caramel          - Blank & Schieberle; Furaneol (HDMF) and maltol as caramel-like key odorants.
  mint/lemon/clove/cinnamon/cucumber/grape/pineapple/coconut/jasmine/violet
                   - classic key-odorant associations for these PLEASANT flavors (terpenoids,
                     phenylpropanoids, green aldehydes, ionones, N-heterocycles, lactones).

OPERATIONAL DEFINITION OF "DELICIOUSNESS" (the scope of this experiment). We restrict the
positive class to the family of IMMEDIATELY PLEASANT flavors dominated by SWEET, CREAMY, FRUITY,
FLORAL, and CITRUS notes: the dessert / beverage / confection / fragrance space (chocolate,
vanilla, caramel, butter, the fruits, mint, floral, citrus, coffee as a roasted-sweet edge).
These are the profiles most people find pleasant on first exposure, and their key odorants have
relatively well-characterized, clean pleasant-receptor associations, so "good" here is a
receptor-verifiable target, not a matter of acquired taste.

We DELIBERATELY EXCLUDE savory, meaty, garlicky, alliaceous, cheesy, and smoky flavors. Many
people genuinely find those delicious, and nothing here claims otherwise; but their pleasantness
is more context- and culture-dependent (acquired rather than immediate), and they are dominated
by sulfur/thiol/pungent chemistry that is harder to tie to a clean pleasant-receptor signal. So
for THIS experiment "deliciousness" operationally means the immediately-pleasant sweet/creamy/
fruity/floral/citrus family above, and results should be read within that scope.

Consequences: esters and lactones are heavily represented on purpose. That is not a sampling
bias, it is real flavor chemistry: the repo's own composition statistics show lactones and esters
carry the highest pleasantness lifts (composition.py), matching the creamy/fruity core of this
family. Sulfur/thiol chemistry still appears where it belongs in a pleasant flavor (coffee's
furfurylthiol, cocoa) rather than as a standalone savory positive.

All SMILES are RDKit-parseable and were verified before inclusion.
"""

REFERENCE_FLAVORS = {
    "chocolate": {
        "3-methylbutanal": "CC(C)CC=O",
        "2-methylbutanal": "CCC(C)C=O",
        "phenylacetaldehyde": "O=CCc1ccccc1",
        "2-phenylethanol": "OCCc1ccccc1",
        "2,3,5-trimethylpyrazine": "Cc1cnc(C)c(C)n1",
        "tetramethylpyrazine": "Cc1nc(C)c(C)nc1C",
        "linalool": "CC(C)=CCCC(C)(O)C=C",
        "2-phenylethyl acetate": "CC(=O)OCCc1ccccc1",
    },
    "vanilla": {
        "vanillin": "O=Cc1ccc(O)c(OC)c1",
        "ethyl vanillin": "CCOc1cc(C=O)ccc1O",
        "guaiacol": "COc1ccccc1O",
        "p-hydroxybenzaldehyde": "O=Cc1ccc(O)cc1",
    },
    "coffee": {
        "furfurylthiol": "SCc1ccco1",
        "guaiacol": "COc1ccccc1O",
        "4-vinylguaiacol": "C=Cc1ccc(O)c(OC)c1",
        "2,3-butanedione": "CC(=O)C(C)=O",
        "furaneol": "CC1=C(O)C(=O)C(C)O1",
    },
    "banana": {
        "isoamyl acetate": "CC(C)CCOC(C)=O",
        "isoamyl alcohol": "CC(C)CCO",
        "isobutyl acetate": "CC(C)COC(C)=O",
    },
    "strawberry": {
        "furaneol": "CC1=C(O)C(=O)C(C)O1",
        "ethyl butyrate": "CCCC(=O)OCC",
        "ethyl hexanoate": "CCCCCC(=O)OCC",
        "methyl cinnamate": "COC(=O)/C=C/c1ccccc1",
        "gamma-decalactone": "CCCCCCC1CCC(=O)O1",
    },
    "rose": {
        "2-phenylethanol": "OCCc1ccccc1",
        "geraniol": "CC(C)=CCC/C(C)=C/CO",
        "citronellol": "CC(CCC=C(C)C)CCO",
        "beta-damascenone": "CC=CC(=O)C1=C(C)C=CCC1(C)C",
    },
    "butter": {  # Schieberle/Grosch, Widder & Schieberle: butter aroma
        "2,3-butanedione": "CC(=O)C(C)=O",
        "delta-decalactone": "CCCCCC1CCCC(=O)O1",
        "butanoic acid": "CCCC(=O)O",
        "gamma-decalactone": "CCCCCCC1CCC(=O)O1",
    },
    "apple": {  # Schieberle et al.: key odorants of apple
        "ethyl 2-methylbutanoate": "CCOC(=O)C(C)CC",
        "hexyl acetate": "CCCCCCOC(C)=O",
        "(E)-2-hexenal": "CC/C=C/C=O",
        "ethyl butanoate": "CCCC(=O)OCC",
    },
    "orange": {  # Buttery et al.: orange volatiles
        "limonene": "CC(=C)C1CCC(C)=CC1",
        "octanal": "CCCCCCCC=O",
        "decanal": "CCCCCCCCCC=O",
        "linalool": "CC(C)=CCCC(C)(O)C=C",
        "ethyl butanoate": "CCCC(=O)OCC",
    },
    "peach": {  # lactone-dominated key-odorant literature
        "gamma-decalactone": "CCCCCCC1CCC(=O)O1",
        "delta-decalactone": "CCCCCC1CCCC(=O)O1",
        "benzaldehyde": "O=Cc1ccccc1",
        "linalool": "CC(C)=CCCC(C)(O)C=C",
        "(Z)-3-hexen-1-ol": "CC/C=C\\CCO",
    },
    "honey": {  # phenylacetaldehyde / 2-phenylethanol honey-aroma studies
        "phenylacetaldehyde": "O=CCc1ccccc1",
        "2-phenylethanol": "OCCc1ccccc1",
        "phenylacetic acid": "OC(=O)Cc1ccccc1",
    },
    "caramel": {  # Blank & Schieberle: Furaneol (HDMF) + maltol as caramel key odorants
        "furaneol": "CC1=C(O)C(=O)C(C)O1",
        "maltol": "Cc1occc1O",
        "2,3-butanedione": "CC(=O)C(C)=O",
    },
    # --- chemotype-diverse additions (avoid biasing "good flavor" toward esters/lactones) ---
    "mint": {  # peppermint oil key odorants (terpenoids)
        "menthol": "CC(C)C1CCC(C)CC1O",
        "menthone": "CC(C)C1CCC(C)CC1=O",
        "1,8-cineole": "CC12CCC(CC1)C(C)(C)O2",
    },
    "lemon": {  # citral-dominated citrus (terpenoids/aldehydes)
        "citral": "CC(C)=CCC/C(C)=C/C=O",
        "limonene": "CC(=C)C1CCC(C)=CC1",
        "citronellal": "O=CCC(C)CCC=C(C)C",
        "nerol": "CC(C)=CCC/C(C)=C\\CO",
    },
    "clove": {  # eugenol-dominated (phenylpropanoids)
        "eugenol": "C=CCc1ccc(O)c(OC)c1",
        "isoeugenol": "C/C=C/c1ccc(O)c(OC)c1",
        "4-methylguaiacol": "COc1cc(C)ccc1O",
    },
    "cinnamon": {  # cinnamaldehyde-dominated (phenylpropanoids)
        "cinnamaldehyde": "O=C/C=C/c1ccccc1",
        "eugenol": "C=CCc1ccc(O)c(OC)c1",
    },
    "cucumber": {  # green C9/C6 aldehydes
        "(E,Z)-2,6-nonadienal": "O=C/C=C/C/C=C\\CCC",
        "(Z)-3-hexenal": "CC/C=C\\CC=O",
        "hexanal": "CCCCCC=O",
    },
    "grape": {  # Concord grape: methyl anthranilate (aromatic amine ester) + esters
        "methyl anthranilate": "COC(=O)c1ccccc1N",
        "ethyl hexanoate": "CCCCCC(=O)OCC",
        "ethyl butanoate": "CCCC(=O)OCC",
    },
    "pineapple": {  # esters + furaneol
        "allyl hexanoate": "CCCCCC(=O)OCC=C",
        "ethyl hexanoate": "CCCCCC(=O)OCC",
        "furaneol": "CC1=C(O)C(=O)C(C)O1",
        "ethyl butanoate": "CCCC(=O)OCC",
    },
    "coconut": {  # lactone-dominated
        "gamma-nonalactone": "CCCCCC1CCC(=O)O1",
        "delta-octalactone": "CCCC1CCCC(=O)O1",
        "gamma-decalactone": "CCCCCCC1CCC(=O)O1",
    },
    "bread": {  # Schieberle 1991: 2-acetyl-1-pyrroline (crust/popcorn) + furan (N-heterocycle/furan)
        "2-acetyl-1-pyrroline": "CC(=O)C1=NCCC1",
        "furfural": "O=Cc1ccco1",
        "2,3-butanedione": "CC(=O)C(C)=O",
    },
    "jasmine": {  # floral: benzyl acetate + indole (N-heterocycle) + cis-jasmone
        "benzyl acetate": "CC(=O)OCc1ccccc1",
        "indole": "c1ccc2[nH]ccc2c1",
        "cis-jasmone": "CC/C=C\\CC1=C(C)CCC1=O",
    },
    "violet": {  # ionone/rose-ketone chemotype
        "beta-ionone": "CC(=O)/C=C/C1=C(C)CCCC1(C)C",
        "beta-damascenone": "CC=CC(=O)C1=C(C)C=CCC1(C)C",
    },
}
