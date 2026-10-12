"""Quantify component-level overlap behind the mixture evaluation (does 'held-out mixture' still
share molecules with training?). Writes results/newslate/mixture_leakage_check.json."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import real_mixture_eval as rme
snitz, ravia, _ = rme.load_pairs()
comps = lambda ps: {c for A, B, _ in ps for m in (A, B) for c in m}
out = {"molecules_snitz": len(comps(snitz)), "molecules_ravia": len(comps(ravia)),
       "molecules_shared_between_datasets": len(comps(snitz) & comps(ravia)),
       "frac_ravia_test_molecules_seen_in_snitz_train": len(comps(snitz) & comps(ravia)) / len(comps(ravia)),
       "frac_snitz_test_molecules_seen_in_ravia_train": len(comps(snitz) & comps(ravia)) / len(comps(snitz))}
for name, ps in (("Snitz", snitz), ("Ravia", ravia)):
    fr = []
    for tr, te in rme.lmo_folds(ps):
        ct, ce = comps(tr), comps(te)
        fr.append(len(ce & ct) / len(ce))
    out[f"{name}_LMO_frac_of_test_pair_molecules_also_in_train_pairs"] = {"mean": float(np.mean(fr)), "per_fold": fr}
json.dump(out, open(os.path.join(HERE, "..", "results", "newslate", "mixture_leakage_check.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
