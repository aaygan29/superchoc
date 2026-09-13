"""Taste / flavor profile layer: map a recipe's molecular chemistry to (a) the five basic
tastes, (b) the chef-description odor vocabulary, and (c) geometric flavor REGIONS, and expose a
"lever" that steers a recipe toward a target profile.

Honest scope. Two different senses of "flavor" are involved and must not be conflated:
  - TASTE (gustation): five qualities, sweet / sour / salty / bitter / umami, transduced on the
    tongue. Most superchoc molecules are ODORANTS, which drive retronasal AROMA, not gustation.
    True salty and umami come from ions / glutamate / nucleotides that are largely outside the
    odorant chemistry here, so those axes are weak by construction and flagged as such.
  - AROMA / FLAVOR descriptors: the ~113 Leffingwell "chef" words (sweet, fruity, green, floral,
    caramellic, roasted, minty, chocolate, ...). These ARE what the molecules determine, and are
    the grounded part of this layer.
So the basic-taste vector is an explicit aroma->taste HEURISTIC (built from real Leffingwell
descriptors + chemical classes), while the descriptor profile and geometric region are grounded
in the molecules' actual odor descriptors.

Geometric flavor regions: each Leffingwell descriptor is placed at the centroid of the molecules
carrying it, in the flavor-space embedding (flavor_geometry). A recipe's position then falls
near some descriptor regions -- that is the chef description mapped geometrically onto flavor
space. The `lever` re-ranks candidate molecules by how much they pull the recipe toward a target
descriptor region, i.e. a control knob on the experienced flavor.

Refs: five basic tastes (Chandrashekar et al. 2006, Nature); Leffingwell odor descriptors (via
Pyrfume); flavor-space geometry as in flavor_geometry.py (Sharpee 2018).
"""

from __future__ import annotations

import numpy as np

import flavor_geometry as fg

# Five basic tastes -> real Leffingwell descriptor names that connote them, plus chemical
# classes (composition.py) that contribute. Weak/near-empty axes are flagged honestly.
TASTE_DESCRIPTORS = {
    "sweet":  ["sweet", "caramellic", "vanilla", "honey", "fruity", "creamy", "coconut"],
    "sour":   ["sour", "fermented", "winey", "citrus", "lemon"],
    "salty":  [],   # no odor descriptor: salty is ionic gustation, outside odorant chemistry
    "bitter": ["phenolic", "burnt", "roasted", "medicinal", "smoky", "coffee", "cocoa"],
    "umami":  ["savory", "brothy", "meaty", "beefy", "broth"],
}
TASTE_CLASSES = {
    "sweet":  {"ester": 0.6, "lactone": 0.8},
    "sour":   {"carboxylic_acid": 1.0},
    "salty":  {},
    "bitter": {"phenol": 0.7, "pyrazine": 0.6, "furan": 0.4},
    "umami":  {"sulfur": 0.3},
}


class TasteProfiler:
    def __init__(self, dim=3, max_mols=2500):
        self.geo = fg.FlavorSpaceGeometry(dim=dim, max_mols=max_mols)
        self.desc = list(self.geo.desc_names)
        self.didx = {d: i for i, d in enumerate(self.desc)}
        # descriptor region centroids in the embedding coordinate space
        V, C = self.geo.V, self.geo.coords
        self.region = {}
        for d, i in self.didx.items():
            mask = V[:, i] > 0
            if mask.sum() >= 3:
                self.region[d] = C[mask].mean(0)

    # -- profile ------------------------------------------------------------------------------
    def _descriptor_profile(self, smiles_list):
        """Mean Leffingwell descriptor profile of the recipe (nearest-molecule for unknowns)."""
        return self.geo._blend_descriptor(smiles_list)

    def basic_tastes(self, smiles_list, comp_vec=None):
        """Heuristic 5-taste vector from descriptor profile (+ optional class fractions)."""
        prof = self._descriptor_profile(smiles_list)
        if prof is None:
            return None
        out = {}
        for taste, descs in TASTE_DESCRIPTORS.items():
            s = sum(prof[self.didx[d]] for d in descs if d in self.didx)
            if comp_vec is not None:
                s += sum(w * comp_vec.get(c, 0.0) for c, w in TASTE_CLASSES[taste].items())
            out[taste] = float(s)
        tot = sum(out.values()) + 1e-9
        return {k: round(v / tot, 3) for k, v in out.items()}

    def profile_strength(self, smiles_list):
        """How 'strong'/focused the flavor is: 1 - normalized entropy of the descriptor profile.
        High = a peaked, characterful profile; low = muddy/washed-out (toward olfactory white)."""
        prof = self._descriptor_profile(smiles_list)
        if prof is None or prof.sum() <= 0:
            return None
        p = prof / prof.sum()
        p = p[p > 0]
        H = -(p * np.log(p)).sum()
        return round(1.0 - H / np.log(len(prof)), 3)

    def flavor_region(self, smiles_list, k=6):
        """Nearest descriptor regions to the recipe's position in the embedding (where it lands)."""
        prof = self._descriptor_profile(smiles_list)
        if prof is None:
            return []
        coord = (prof - self.geo._mean) @ self.geo.components.T
        d = {name: float(np.linalg.norm(coord - c)) for name, c in self.region.items()}
        return [name for name, _ in sorted(d.items(), key=lambda kv: kv[1])[:k]]

    def profile(self, smiles_list, comp_vec=None):
        prof = self._descriptor_profile(smiles_list)
        bt = self.basic_tastes(smiles_list, comp_vec)
        top_desc = []
        if prof is not None:
            idx = np.argsort(prof)[::-1]
            top_desc = [(self.desc[i], round(float(prof[i]), 2)) for i in idx[:6] if prof[i] > 0]
        return {"basic_tastes": bt, "dominant_taste": (max(bt, key=bt.get) if bt else None),
                "top_descriptors": top_desc, "profile_strength": self.profile_strength(smiles_list),
                "flavor_region": self.flavor_region(smiles_list),
                "note": "basic tastes are an aroma->taste heuristic; salty/umami are weak "
                        "because they are ionic gustation outside odorant chemistry"}

    # -- lever (control knob) -----------------------------------------------------------------
    def lever(self, recipe_smiles, target_descriptor, candidate_smiles, k=8):
        """Rank candidate molecules by how much ADDING each moves the recipe's profile toward
        the target descriptor region. Returns the best candidates (the control on flavor)."""
        if target_descriptor not in self.region:
            return {"error": f"unknown/empty descriptor region: {target_descriptor}"}
        target = self.region[target_descriptor]
        base = self._descriptor_profile(recipe_smiles)
        if base is None:
            return {"error": "recipe profile unavailable"}
        base_coord = (base - self.geo._mean) @ self.geo.components.T
        base_d = float(np.linalg.norm(base_coord - target))
        scored = []
        for s in candidate_smiles:
            prof = self._descriptor_profile(recipe_smiles + [s])
            if prof is None:
                continue
            coord = (prof - self.geo._mean) @ self.geo.components.T
            move = base_d - float(np.linalg.norm(coord - target))  # positive = moves toward
            scored.append((s, round(move, 4)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return {"target": target_descriptor, "base_distance": round(base_d, 3),
                "suggestions": [{"smiles": s, "pull_toward_target": m} for s, m in scored[:k]]}


if __name__ == "__main__":
    import json
    from reference_flavors import REFERENCE_FLAVORS
    tp = TasteProfiler()
    for flavor in ["chocolate", "vanilla", "strawberry"]:
        smis = list(REFERENCE_FLAVORS[flavor].values())
        p = tp.profile(smis)
        print(f"{flavor}: dominant_taste={p['dominant_taste']} strength={p['profile_strength']} "
              f"tastes={p['basic_tastes']}")
        print(f"   top notes: {p['top_descriptors'][:5]}")
        print(f"   region: {p['flavor_region']}")
