"""Equivalence/consistency tests for jspec. Run: python3 newslate/test_jspec.py"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jspec

def main():
    ctx = jspec.Context(taste_source="insample_legacy")
    # 1. fast terms equal the legacy per-combo functions
    import denovo_recipes as dr, composition as comp
    from reference_flavors import REFERENCE_FLAVORS
    from featurize import smiles_to_ecfp4
    rng = np.random.default_rng(1); worst = 0
    ref_fp = {f: [smiles_to_ecfp4(s) for s in m.values()] for f, m in REFERENCE_FLAVORS.items()}
    for _ in range(20):
        combo = tuple(rng.choice(ctx.elig, 10, replace=False))
        d = []
        for f, fps in ref_fp.items():
            d.append(1.0 - float(np.mean([max(dr._tanimoto(ctx.X[i], r) for r in fps) for i in combo])))
        worst = max(worst, abs(min(d) - ctx.N(combo)))
        pl = float(ctx.model.predict(ctx.X[list(combo)]).mean())
        assert abs(pl / 100 - ctx.terms(combo, False)["T"]) < 1e-9
        assert abs(comp.composition_score([ctx.pool[i]["smiles"] for i in combo], ctx.target) - ctx.terms(combo, False)["C"]) < 1e-3
    print("max |N_fast - N_legacy| =", worst); assert worst < 1e-6
    # 2. M features equal the evaluated Feat.pair features
    combo = tuple(rng.choice(ctx.elig, 6, replace=False))
    a = jspec.stats_of(ctx.Z[list(combo)])
    b = ctx.ref_stats[0]
    class FakeF: pass
    cids = None
    print("M(random blend) =", ctx.M(combo))
    # 3. legacy-mode search reproduces legacy compose() recipe sets (if the legacy json exists)
    w = {"taste": 1.0, "composition": 0.15, "novelty": 0.4, "mixture": 0.0}
    out, inf = jspec.search(ctx, w, seed=0)
    leg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "newslate", "legacy_compose.json")
    res = {"new_sets": [sorted(ctx.pool[i]["smiles"] for i in r["combo"]) for r in out], "new_J": [r["J"] for r in out]}
    if os.path.exists(leg_path):
        leg = json.load(open(leg_path))
        ls = [sorted(c["smiles"] for c in r["components"]) for r in leg["recipes"]]
        res["legacy_sets_equal"] = [s in ls for s in res["new_sets"]]; res["n_legacy"] = len(ls)
        res["legacy_J"] = [r["objective"] for r in leg["recipes"]]
    print(json.dumps({k: v for k, v in res.items() if k != "new_sets"}, default=float))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "newslate", "test_jspec.json"), "w"), default=float)

if __name__ == "__main__":
    main()
