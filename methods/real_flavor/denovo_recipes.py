"""Statistically-composed recipes: use the learned best-composition target plus the goodness
model to build the best new recipes, drawing from BOTH known-safe molecules and the de-novo
"lego" molecules.

Ties the whole program together:
  - composition.py gives the target class balance (favor lactone/ether/ester/aromatic/phenol;
    avoid sulfur/pyrazine/acid/furan/amine), learned from real pleasantness data;
  - the goodness model gives predicted pleasantness;
  - reference_flavors gives chemical novelty;
  - safety.py gates toxicity;
  - de-novo molecules (results/lego_denovo.json) expand the palette beyond known compounds.

Objective per recipe:
    J = w_taste*mean_pleasantness/100 + w_comp*composition_score + w_novel*chem_novelty
Empirical check: composition_score correlates with true mean pleasantness across random
blends (validates the "statistically best composition" claim). Lean backbone: FlavorMath.lean
(composition can't beat the best class additively -> interactions needed).
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from goodness_real import RealGoodnessModel
from safety import screen
import composition as comp
from reference_flavors import REFERENCE_FLAVORS
import flavor_geometry as fg

_HERE = os.path.dirname(__file__)


def _tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def _prep():
    df = build_keller_pleasantness().reset_index(drop=True)
    pool, X = [], []
    for _, r in df.iterrows():
        try:
            fp = smiles_to_ecfp4(r["smiles"])
        except Exception:
            continue
        X.append(fp)
        pool.append({"name": r["name"], "smiles": r["smiles"], "source": "known",
                     "pleasantness": float(r["pleasantness"]),
                     "safety": screen(r["smiles"])["verdict"]})
    X = np.vstack(X)
    model = RealGoodnessModel(seed=0).fit(X, np.array([p["pleasantness"] for p in pool]))

    stats = comp.class_pleasantness_stats(df)
    target = comp.target_composition(stats)

    # add de-novo molecules from the lego run if available
    lego_path = os.path.join(_HERE, "results", "lego_denovo.json")
    if os.path.exists(lego_path):
        lego = json.load(open(lego_path))
        for e in lego.get("candidates", [])[:40]:
            if e.get("safety") == "OK":
                try:
                    fp = smiles_to_ecfp4(e["smiles"])
                except Exception:
                    continue
                X = np.vstack([X, fp])
                pool.append({"name": f"denovo:{e['smiles']}", "smiles": e["smiles"],
                             "source": "de-novo", "pleasantness": float(e["predicted_pleasantness"]),
                             "safety": "OK"})

    ref_fp = {f: [smiles_to_ecfp4(s) for s in mols.values()] for f, mols in REFERENCE_FLAVORS.items()}
    return df, pool, X, model, stats, target, ref_fp


def validate_composition(df, target, n=1500, seed=0):
    """Does composition_score predict true mean pleasantness across random blends?"""
    rng = np.random.default_rng(seed)
    rows = [(r["smiles"], float(r["pleasantness"])) for _, r in df.iterrows()]
    cs, pl = [], []
    for _ in range(n):
        idx = rng.choice(len(rows), size=6, replace=False)
        smis = [rows[i][0] for i in idx]
        cs.append(comp.composition_score(smis, target))
        pl.append(float(np.mean([rows[i][1] for i in idx])))
    return round(float(spearmanr(cs, pl).correlation), 4)


def _locate_recipes(recipes_out):
    """Attach a flavor-space position to each recipe via the geometry layer (locate): its map
    coordinate, nearest known flavors, and a novelty percentile vs same-size random blends.
    This is the evaluation/hypothesis layer that connects the (validated) flavor-space
    mathematics to each generated recipe. Guarded: a geometry failure leaves recipes intact."""
    try:
        geo = fg.FlavorSpaceGeometry(dim=3)
    except Exception as e:
        for r in recipes_out:
            r["flavor_space"] = {"error": f"geometry unavailable: {type(e).__name__}"}
        return recipes_out
    for r in recipes_out:
        try:
            r["flavor_space"] = geo.locate([c["smiles"] for c in r["components"]])
        except Exception as e:
            r["flavor_space"] = {"error": f"locate failed: {type(e).__name__}"}
    return recipes_out


def compose(seed=0, n_components=10, n_recipes=5, iters=500,
            w_taste=1.0, w_comp=0.15, w_novel=0.4, chem_novelty_min=0.5,
            with_geometry=True):
    df, pool, X, model, stats, target, ref_fp = _prep()
    elig = np.array([i for i, p in enumerate(pool) if p["safety"] == "OK"])
    rng = np.random.default_rng(seed)

    def chem_novelty(combo):
        d = []
        for f, fps in ref_fp.items():
            d.append(1.0 - float(np.mean([max(_tanimoto(X[i], r) for r in fps) for i in combo])))
        return min(d)

    def obj(combo):
        pl = float(model.predict(X[list(combo)]).mean())
        smis = [pool[i]["smiles"] for i in combo]
        cscore = comp.composition_score(smis, target)
        nov = chem_novelty(combo)
        return w_taste * pl / 100 + w_comp * cscore + w_novel * nov, pl, cscore, nov

    recipes = []
    for _ in range(n_recipes):
        combo = tuple(rng.choice(elig, size=n_components, replace=False))
        t = 0
        while chem_novelty(combo) < chem_novelty_min and t < 100:
            combo = tuple(rng.choice(elig, size=n_components, replace=False)); t += 1
        bestJ = obj(combo)[0]
        for _ in range(iters):
            cand = list(combo); cand[rng.integers(n_components)] = int(rng.choice(elig))
            if len(set(cand)) < n_components:
                continue
            cand = tuple(cand)
            if chem_novelty(cand) < chem_novelty_min:
                continue
            J = obj(cand)[0]
            if J > bestJ:
                combo, bestJ = cand, J
        if frozenset(combo) not in {frozenset(c) for c, _ in recipes}:
            recipes.append((combo, bestJ))

    recipes.sort(key=lambda t: t[1], reverse=True)
    out = []
    for rank, (combo, J) in enumerate(recipes, 1):
        _, pl, cscore, nov = obj(combo)
        smis = [pool[i]["smiles"] for i in combo]
        cv = comp.composition_vector(smis)
        cv_top = sorted(((k, v) for k, v in cv.items() if v > 0), key=lambda kv: kv[1], reverse=True)[:6]
        comps = sorted(({"molecule": pool[i]["name"], "smiles": pool[i]["smiles"],
                         "source": pool[i]["source"], "pleasantness": round(pool[i]["pleasantness"], 1),
                         "classes": sorted(comp.classify(pool[i]["smiles"]))} for i in combo),
                       key=lambda c: c["pleasantness"], reverse=True)
        out.append({"rank": rank, "objective": round(J, 4),
                    "predicted_mean_pleasantness": round(pl, 1),
                    "composition_score": cscore, "chem_novelty": round(nov, 3),
                    "n_denovo_components": sum(1 for c in comps if c["source"] == "de-novo"),
                    "composition_top_classes": [(k, round(v, 2)) for k, v in cv_top],
                    "components": comps})
    if with_geometry:
        out = _locate_recipes(out)
    return {"target_composition": target,
            "composition_predicts_pleasantness_spearman": validate_composition(df, target, seed=seed),
            "class_pleasantness_stats": stats, "recipes": out}


if __name__ == "__main__":
    d = compose()
    print("composition->pleasantness Spearman:", d["composition_predicts_pleasantness_spearman"])
    print("target favor:", d["target_composition"]["favor"], "avoid:", d["target_composition"]["avoid"])
    for r in d["recipes"]:
        fsp = r.get("flavor_space") or {}
        print(f"  #{r['rank']} pl={r['predicted_mean_pleasantness']} comp={r['composition_score']} "
              f"nov={r['chem_novelty']} denovo={r['n_denovo_components']} "
              f"top={[k for k,_ in r['composition_top_classes'][:3]]}")
        print(f"     flavor-space: novelty_pct={fsp.get('novelty_percentile')} "
              f"near={fsp.get('nearest_known_flavors')} -> {fsp.get('interpretation', fsp.get('error'))}")
