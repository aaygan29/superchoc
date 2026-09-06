"""Reference flavors as their published key-odorant sets (SMILES).

Used to (a) anchor "novelty of the combination": a candidate recipe should be distinct
from every known flavor here, not just from cocoa; and (b) sanity-check that a candidate
is not simply a rediscovery of vanilla/coffee/etc.

These are representative key aroma compounds from the flavor-chemistry literature (e.g.
Frauendorfer & Schieberle for cocoa; standard key-odorant lists for the others). They are
NOT exhaustive formulas; they are enough to place a candidate relative to known flavors.
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
}
