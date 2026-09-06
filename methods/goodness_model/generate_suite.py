"""Generate a ranked SUITE of candidate 'super-chocolate' flavor combinations.

This is the deliverable: a set of molecule combos, each predicted to be as good as or
better than a chocolate reference while being novel (chemically distinct from it), ranked
so a chemistry lab or a molecular gastronomist could pick the top few to actually make
and taste. The loop composes combos, iteratively refines each by single-molecule swaps
(hill-climb on predicted goodness), and keeps only novelty-valid ones.

IMPORTANT: run as-is this operates on the SYNTHETIC flavor space, so "molecules" are pool
indices, not real compounds, and scores are demonstration values, not real taste. To make
a real suite: build a molecule table (SMILES + name) and a goodness model trained on real
labels (see data/ and featurize.py), then pass a molecule_table mapping index -> {name,
smiles}. The banner in the output makes the synthetic-vs-real status explicit so no one
mistakes a demo suite for a validated recipe.
"""

from __future__ import annotations

import json
import sys

from flavor_space import FlavorSpace
from goodness_model import evaluate_model
from compose import compose_combos


def generate_suite(seed=0, k=4, n_candidates=8, novelty_min=0.2, iters=400,
                   molecule_table=None):
    space = FlavorSpace(seed=seed)
    model, r2 = evaluate_model(space, n_train=400, seed=seed)
    ref = space.chocolate_reference(k=k)

    cands = compose_combos(space, model, k=k, n_combos=n_candidates,
                           novelty_min=novelty_min, ref_embedding=ref["embedding"],
                           iters=iters, seed=seed)
    # Rank by predicted goodness (what a real deployment would order by), descending.
    cands.sort(key=lambda c: c["predicted_goodness"], reverse=True)

    def render_members(combo):
        if molecule_table is None:
            return [f"mol#{i}" for i in combo]
        return [molecule_table.get(i, {}).get("name", f"mol#{i}") for i in combo]

    suite = {
        "is_synthetic_demo": molecule_table is None,
        "banner": ("SYNTHETIC DEMO: members are pool indices, scores are demonstration "
                   "values. Swap in a real molecule table + real goodness model for a "
                   "lab-testable suite." if molecule_table is None
                   else "REAL molecules from provided table."),
        "goodness_model_r2": round(float(r2), 4),
        "chocolate_reference_true_goodness": round(float(ref["value"]), 4),
        "candidates": [
            {
                "rank": n + 1,
                "members": render_members(c["combo"]),
                "member_indices": [int(i) for i in c["combo"]],
                "predicted_goodness": round(float(c["predicted_goodness"]), 4),
                "true_goodness_oracle": round(float(c["true_goodness"]), 4),
                "novelty_from_chocolate": (round(float(c["novelty_from_ref"]), 4)
                                           if c["novelty_from_ref"] is not None else None),
                "beats_chocolate": bool(c["true_goodness"] >= ref["value"]),
            }
            for n, c in enumerate(cands)
        ],
    }
    return suite


def to_markdown(suite) -> str:
    lines = [
        "# Candidate super-chocolate suite",
        "",
        f"> {suite['banner']}",
        "",
        f"Goodness model held-out R2: {suite['goodness_model_r2']}. "
        f"Chocolate reference goodness: {suite['chocolate_reference_true_goodness']}.",
        "",
        "| rank | members | predicted | true (oracle) | novelty vs choc | beats choc |",
        "|---|---|---|---|---|---|",
    ]
    for c in suite["candidates"]:
        lines.append(
            f"| {c['rank']} | {', '.join(c['members'])} | {c['predicted_goodness']} | "
            f"{c['true_goodness_oracle']} | {c['novelty_from_chocolate']} | "
            f"{'yes' if c['beats_chocolate'] else 'no'} |"
        )
    lines += ["", "_Members are the molecules to combine. In a real run each is a named "
              "compound with a SMILES and a synthesis/sourcing note for the lab._"]
    return "\n".join(lines)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    suite = generate_suite(seed=seed)
    print(json.dumps(suite, indent=2))
