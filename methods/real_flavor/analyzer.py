"""SuperchocAnalyzer: the one overarching analyzer that synthesizes the whole pipeline.

It takes chemical structures, per-molecule predictions, known reference flavors, and the
geometric flavor-space representation, and returns internally-evaluated predictions for NEW
flavor combos/recipes that are novel and theorized to taste good, each with a plain-language
description of what the taste is meant to be like.

It unifies every component built in this repo:
  - goodness_real  : per-molecule human pleasantness (real Keller labels)
  - safety         : food-appropriate toxicity gate
  - lego_assembly  : de-novo molecules (for structural novelty)
  - flavor_geometry: hyperbolic flavor-space position + perceptual novelty percentile
  - mixtures       : Snitz/Ravia-validated mixture perceptual distance + Scheffe ratio optimizer
  - recipe_classifier: recipe-level P(good flavor) (leave-one-flavor-out AUC 0.69)
  - taste_profile  : five basic tastes + chef-descriptor flavor regions

Everything is INTERNAL evaluation on validated / externally-grounded models plus a safety gate.
Nothing is human-tasted; outputs are ranked hypotheses for a chemist or chef (see HANDOFF.md).
"""

from __future__ import annotations

import json
import os

import numpy as np

from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
from safety import screen
import composition as comp
import mixtures as mx
import recipe_classifier as rc
import taste_profile as tp
from reference_flavors import REFERENCE_FLAVORS

_HERE = os.path.dirname(__file__)


def _describe(profile, nearest_flavor, chem_nov, perc_nov, good_p):
    """Plain-language description of what the recipe is theorized to taste like."""
    bt = profile.get("basic_tastes") or {}
    dom = profile.get("dominant_taste")
    notes = [d for d, _ in (profile.get("top_descriptors") or [])][:3]
    strg = profile.get("profile_strength") or 0.0
    intensity = "bold and focused" if strg >= 0.5 else ("distinct" if strg >= 0.35 else "delicate")
    taste_bits = [t for t, v in sorted(bt.items(), key=lambda kv: -kv[1]) if v >= 0.2]
    taste_str = " and ".join(taste_bits) if taste_bits else (dom or "mild")
    note_str = ", ".join(notes) if notes else "clean"
    rel = (f"closest to {nearest_flavor} but shifted into a {'/'.join((profile.get('flavor_region') or [''])[:2])} region"
           if nearest_flavor else "in an unfamiliar region of flavor space")
    return (f"A {intensity}, predominantly {taste_str} flavor led by {note_str} notes; "
            f"perceptually {rel}. Predicted good-flavor {good_p:.2f}; "
            f"chemical novelty {chem_nov:.2f}, perceptual novelty {perc_nov:.2f}.")


class SuperchocAnalyzer:
    def __init__(self, allow_denovo=True, pool_size=110):
        df = build_keller_pleasantness().reset_index(drop=True)
        self.pl = {r["smiles"]: float(r["pleasantness"]) for _, r in df.iterrows()}
        self.name = {r["smiles"]: r["name"] for _, r in df.iterrows()}
        pool = [(r["name"], r["smiles"]) for _, r in df.iterrows()
                if screen(r["smiles"])["verdict"] == "OK"]
        pool.sort(key=lambda ns: self.pl.get(ns[1], 0), reverse=True)
        self.known_pool = [s for _, s in pool[:pool_size]]
        self.denovo_pool = []
        if allow_denovo:
            p = os.path.join(_HERE, "results", "lego_denovo.json")
            if os.path.exists(p):
                for e in json.load(open(p)).get("candidates", []):
                    if e.get("safety") == "OK":
                        self.denovo_pool.append(e["smiles"])
                        self.name[e["smiles"]] = f"denovo:{e['smiles']}"
        self.clf = rc.RecipeClassifier().fit()
        self.profiler = tp.TasteProfiler()
        self.geo = self.profiler.geo  # reuse the loaded flavor-space geometry
        self.ref_mix = {f: list(m.values()) for f, m in REFERENCE_FLAVORS.items()}
        ref_all = [s for ms in self.ref_mix.values() for s in ms]
        self.ref_all = ref_all
        self.space = mx.MixtureSpace(ref_all + self.known_pool + self.denovo_pool)
        self.ref_fps = {f: [smiles_to_ecfp4(s) for s in ms] for f, ms in self.ref_mix.items()}
        self.target = comp.target_composition(comp.class_pleasantness_stats(df))
        # precompute per-molecule predicted pleasantness for the whole (already-safe) pool once
        allpool = self.known_pool + self.denovo_pool
        Xp = np.vstack([smiles_to_ecfp4(s) for s in allpool])
        preds = self.clf.ctx["model"].predict(Xp)
        self.mol_pl = {s: float(preds[i]) for i, s in enumerate(allpool)}

    # -- molecule-level analysis --------------------------------------------------------------
    def analyze_structure(self, smiles):
        try:
            fp = smiles_to_ecfp4(smiles)
        except Exception:
            return {"smiles": smiles, "error": "unparseable"}
        loc = self.geo.locate([smiles])
        return {"smiles": smiles, "name": self.name.get(smiles, smiles),
                "predicted_pleasantness": round(float(self.clf.ctx["model"].predict(fp.reshape(1, -1))[0]), 1),
                "safety": screen(smiles)["verdict"], "classes": sorted(comp.classify(smiles)),
                "flavor_space_novelty_percentile": (loc or {}).get("novelty_percentile"),
                "nearest_known_flavors": (loc or {}).get("nearest_known_flavors")}

    # -- novelty helpers ----------------------------------------------------------------------
    def _chem_novelty(self, smis):
        """1 - max mean-Tanimoto similarity to any reference flavor (min over flavors)."""
        fps = [smiles_to_ecfp4(s) for s in smis]
        d = []
        for _, rfps in self.ref_fps.items():
            sims = [max((self._tanimoto(f, r) for r in rfps), default=0.0) for f in fps]
            d.append(1.0 - float(np.mean(sims)))
        return min(d) if d else 0.0

    @staticmethod
    def _tanimoto(a, b):
        a = a.astype(bool); b = b.astype(bool)
        i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
        return float(i / u) if u else 0.0

    def _perceptual_novelty(self, smis):
        loc = self.geo.locate(smis)
        return (loc or {}).get("novelty_percentile", 0.0)

    def _evoked_flavor(self, smiles):
        """The flavor a single molecule is known/predicted to evoke: its top odor descriptors
        (predicted for de-novo molecules via nearest-molecule descriptors)."""
        prof = self.geo._blend_descriptor([smiles])
        if prof is None:
            return []
        idx = np.argsort(prof)[::-1]
        return [self.geo.desc_names[i] for i in idx[:3] if prof[i] > 0]

    # -- recipe design ------------------------------------------------------------------------
    def superchocolate(self, **kw):
        """Build toward chocolate but push for higher predicted good-flavor + novelty: the
        theorized 'superchocolate' (a recipe near chocolate in flavor space, scored better)."""
        kw.setdefault("target_flavor", "chocolate")
        kw.setdefault("novelty_weight", 0.35)
        return self.design(**kw)

    def design(self, n_recipes=5, n_components=6, novelty_weight=0.6, target_flavor=None,
               target_weight=0.8, n_candidates=2500, keep_t1=300, keep_t2=40, seed=0):
        """Compute-efficient cascade: generate many candidates, then spend budget only on the
        survivors of each cheap filter (coarse-to-fine).

          Tier 0  safety gate (trivial): drop any recipe with a non-OK component.
          Tier 1  cheap score: mean component pleasantness + composition score (memoized ECFP,
                  no geometry/classifier). Keep the top `keep_t1`.
          Tier 2  medium score: recipe classifier P(good flavor) + perceptual-distance novelty.
                  Keep the top `keep_t2`.
          Tier 3  expensive: full taste profile + Scheffe ratio optimization, only on the
                  finalists -> ranked `n_recipes`.
        """
        rng = np.random.default_rng(seed)
        pool = self.known_pool + self.denovo_pool
        target = self.target

        # generate unique candidate recipes
        cand, seen = [], set()
        while len(cand) < n_candidates and len(seen) < n_candidates * 3:
            combo = tuple(sorted(rng.choice(pool, size=n_components, replace=False)))
            if combo in seen:
                continue
            seen.add(combo)
            cand.append(list(combo))

        # Tier 0: safety. The pool is already safety-filtered in __init__, so every candidate
        # passes by construction; the tier is kept explicit (and would gate a mixed pool).
        t0 = cand

        # Tier 1: cheap pleasantness (precomputed per-molecule) + composition. No geometry,
        # no classifier, no per-candidate model inference.
        def cheap(smis):
            pl = float(np.mean([self.mol_pl.get(s, 50.0) for s in smis]))
            cscore = comp.composition_score(smis, target) if target else 0.0
            return pl / 100.0 + 0.3 * cscore
        t1 = sorted(t0, key=cheap, reverse=True)[:keep_t1]

        # optional: steer toward a named target flavor (e.g. chocolate -> superchocolate)
        tgt = self.ref_mix.get(target_flavor) if target_flavor else None

        # Tier 2: classifier P(good) + perceptual novelty distance (+ proximity to target flavor)
        def medium(smis):
            gp = self.clf.score(smis) or 0.0
            dist = self.space.distance(smis, self.ref_all) or 0.0
            s = gp + novelty_weight * (dist / 3.14159)
            if tgt is not None:
                dt = self.space.distance(smis, tgt)
                if dt is not None:
                    s += target_weight * (1.0 - dt / 3.14159)  # closer to target = better
            return s
        t2 = sorted(t1, key=medium, reverse=True)[:keep_t2]

        # Tier 3: full package (expensive) on finalists, then rank
        packaged = [self._package(c) for c in t2]

        def final_rank(p):
            s = p["good_flavor_probability"] + novelty_weight * p["perceptual_novelty_percentile"]
            if tgt is not None:
                dt = p["perceptual_distance_to_known_flavors"].get(target_flavor)
                if dt is not None:
                    s += target_weight * (1.0 - dt / 3.14159)
            return s
        packaged.sort(key=final_rank, reverse=True)
        self._funnel = {"generated": len(cand), "tier0_safe": len(t0),
                        "tier1_kept": len(t1), "tier2_kept": len(t2), "final": n_recipes}
        return packaged[:n_recipes]

    def _package(self, combo):
        gp = float(self.clf.score(combo) or 0.0)
        prof = self.profiler.profile(combo, comp_vec=comp.composition_vector(combo))
        chem_nov = float(self._chem_novelty(combo))
        perc_nov = float(self._perceptual_novelty(combo))
        opt = mx.optimize_ratios(combo, lambda w: float(self.clf.score(combo, weights=w)))
        dists = {f: self.space.distance(combo, ms) for f, ms in self.ref_mix.items()}
        dists = {f: round(d, 3) for f, d in dists.items() if d is not None}
        nearest = min(dists, key=dists.get) if dists else None
        ratios = opt["best_proportions"]
        components = [{"molecule": self.name.get(s, s), "smiles": s,
                       "proportion": ratios[i] if i < len(ratios) else round(1/len(combo), 3),
                       "source": "de-novo" if s in self.denovo_pool else "known",
                       "pleasantness": round(self.pl.get(s, float("nan")), 1)
                       if s in self.pl else None,
                       "classes": sorted(comp.classify(s)),
                       # each component's intended/known evoked flavor (predicted for de-novo
                       # via nearest-molecule odor descriptors) -> the flavor "type" we build from
                       "evoked_flavor": self._evoked_flavor(s)}
                      for i, s in enumerate(combo)]
        return {
            "good_flavor_probability": round(gp, 3),
            "chemical_novelty": round(chem_nov, 3),
            "perceptual_novelty_percentile": round(perc_nov, 3),
            "profile_strength": prof["profile_strength"],
            "taste_description": _describe(prof, nearest, chem_nov, perc_nov, gp),
            "taste_profile": {"basic_tastes": prof["basic_tastes"],
                              "dominant_taste": prof["dominant_taste"],
                              "top_descriptors": prof["top_descriptors"],
                              "flavor_region": prof["flavor_region"]},
            "optimal_levels": {"proportions": ratios, "interior_optimum": opt["interior_optimum"],
                               "reading": opt["reading"]},
            "perceptual_distance_to_known_flavors": dists, "nearest_known_flavor": nearest,
            "n_denovo_components": sum(1 for c in components if c["source"] == "de-novo"),
            "components": components,
        }

    def report(self, **kw):
        recipes = self.design(**kw)
        out = {"recipes": recipes,
               "cascade_funnel": getattr(self, "_funnel", None),
               "validation": {"recipe_classifier_loo_auc": 0.69,
                              "mixture_distance_snitz_spearman": -0.49,
                              "mixture_distance_ravia_spearman": -0.245},
               "note": ("Internally evaluated on validated/externally-grounded models + safety "
                        "gate. Not human-tasted; ranked hypotheses for a chemist/chef (HANDOFF.md).")}
        return out


if __name__ == "__main__":
    A = SuperchocAnalyzer()
    rep = A.report(n_recipes=5)
    print("cascade funnel:", rep["cascade_funnel"])
    for i, r in enumerate(rep["recipes"], 1):
        print(f"\n#{i}  good={r['good_flavor_probability']} chemNov={r['chemical_novelty']} "
              f"percNov={r['perceptual_novelty_percentile']} denovo={r['n_denovo_components']}/"
              f"{len(r['components'])}")
        print(f"    {r['taste_description']}")
    json.dump(rep, open(os.path.join(_HERE, "results", "superchoc_analysis.json"), "w"), indent=2)

    # superchocolate: build toward chocolate, scored better
    sc = A.superchocolate(n_recipes=3)
    print("\n=== SUPERCHOCOLATE (target=chocolate) ===")
    top = sc[0]
    print(top["taste_description"])
    print("component evoked flavors:")
    for c in top["components"]:
        print(f"  {c['proportion']:.3f}  {c['molecule'][:34]:34s} evokes {c['evoked_flavor']}")
    json.dump({"superchocolate": sc}, open(os.path.join(_HERE, "results", "superchocolate.json"), "w"), indent=2)
    print("\nwrote results/superchoc_analysis.json + results/superchocolate.json")
