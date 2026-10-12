"""Molecule-DISJOINT cross-dataset transfer: train only on pairs whose mixtures contain no molecule that
appears in the test dataset. Strictest leakage-free test of set vs mean-pool. Appends to
results/newslate/mixture_disjoint_transfer.json."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import real_mixture_eval as rme
snitz, ravia, cid2smi = rme.load_pairs(); F = rme.Feat(cid2smi)
comps = lambda ps: {c for A, B, _ in ps for m in (A, B) for c in m}
out = {}
for name, tr, te in (("train_Snitz_disjoint_test_Ravia", snitz, ravia), ("train_Ravia_disjoint_test_Snitz", ravia, snitz)):
    bad = comps(te)
    trd = [p for p in tr if not (set(p[0]) | set(p[1])) & bad]
    if len(trd) < 30: out[name] = {"n_train_pairs": len(trd), "skipped": "too few pairs"}; continue
    r, _, _ = rme.run_protocol(F, trd, te, [0, 1, 2, 3, 4], name); out[name] = r
    print(name, r["n_train_pairs"], {k: round(v, 3) for k, v in r["spearman"].items()}, round(r["set_minus_meanpool"], 3), r["set_minus_meanpool_ci95"])
json.dump(out, open(os.path.join(HERE, "..", "results", "newslate", "mixture_disjoint_transfer.json"), "w"), indent=2, default=float)
