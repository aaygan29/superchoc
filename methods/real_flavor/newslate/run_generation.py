"""NEWSLATE (b)+(c): rerun generation with the mixture term wired into J, and the 2^4 weight ablation.
  python3 newslate/run_generation.py run      # compute (resumable; writes results/newslate/generation_raw.json)
  python3 newslate/run_generation.py analyze  # statistics + figures/fig1_weight_ablation.pdf
"""
import itertools, json, multiprocessing as mp, os, sys, time
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
RAW = os.path.join(OUT, "generation_raw.json")
SPEC = json.load(open(os.path.join(HERE, "configs", "J_spec.json")))
BASE = SPEC["weights"]; TERMS = ["taste", "composition", "novelty", "mixture"]
_ctx = {}


def tasks():
    t = []
    for bits in itertools.product([0, 1], repeat=4):
        for seed in range(5):
            t.append({"cfg": "".join(map(str, bits)), "bits": bits, "seed": seed, "taste_source": "oof", "pool": "all"})
    for seed in range(5, 10):
        for bits in [(1, 1, 1, 0), (1, 1, 1, 1)]:
            t.append({"cfg": "".join(map(str, bits)), "bits": bits, "seed": seed, "taste_source": "oof", "pool": "all"})
    for seed in range(10):
        t.append({"cfg": "legacy_insample", "bits": (1, 1, 1, 0), "seed": seed, "taste_source": "insample_legacy", "pool": "all"})
    for seed in range(5):
        t.append({"cfg": "full_known_only", "bits": (1, 1, 1, 1), "seed": seed, "taste_source": "oof", "pool": "known"})
        t.append({"cfg": "additive_known_only", "bits": (1, 1, 1, 0), "seed": seed, "taste_source": "oof", "pool": "known"})
    for t_ in t: t_["key"] = f"{t_['cfg']}|{t_['seed']}"
    return t


def _init():
    import jspec
    _ctx["oof"] = jspec.Context("oof")
    c = _ctx["oof"]
    _ctx["leg"] = c  # share heavy objects; only the taste vector differs
    _ctx["jspec"] = jspec


def _work(t):
    js = _ctx["jspec"]; c = _ctx["oof"]
    c.taste = c.taste_oof if t["taste_source"] == "oof" else c.taste_full
    w = {k: BASE[k] * b for k, b in zip(TERMS, t["bits"])}
    elig = c.elig if t["pool"] == "all" else np.array([i for i in c.elig if c.is_known[i]])
    t0 = time.time()
    recs, inf = js.search(c, w, seed=t["seed"], elig=elig)
    # report terms with the OOF taste for comparability across configs
    c.taste = c.taste_oof
    for r in recs:
        r["terms_oof"] = {**c.terms(r["combo"], need_M=True)}
    return {**{k: t[k] for k in ("key", "cfg", "seed", "pool", "taste_source")}, "w": w, "recipes": recs,
            "infeasible_init": inf, "secs": time.time() - t0}


def run():
    done = {}
    if os.path.exists(RAW): done = {r["key"]: r for r in json.load(open(RAW))}
    todo = [t for t in tasks() if t["key"] not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with mp.Pool(6, initializer=_init) as pool:
        for i, r in enumerate(tqdm(pool.imap_unordered(_work, todo), total=len(todo), desc="generation")):
            done[r["key"]] = r
            if i % 5 == 0 or i == len(todo) - 1:
                json.dump(list(done.values()), open(RAW, "w"), default=float)
    json.dump(list(done.values()), open(RAW, "w"), default=float)


if __name__ == "__main__":
    if sys.argv[1] == "run": run()
    else:
        import analyze_generation; analyze_generation.main()
