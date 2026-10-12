"""Figure 2: set-pooled (non-additive) vs mean-pooled (additive) mixture model, synthetic oracle beside REAL held-out human data."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", "newslate")
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "figures"))
d = json.load(open(os.path.join(R, "real_mixture_eval.json"))); dj = json.load(open(os.path.join(R, "mixture_disjoint_transfer.json")))
syn = d["synthetic_oracle"]; ci = lambda m, s: 1.96 * s / np.sqrt(5)
panels = []
panels.append(("Synthetic, built non-additive\n(0.8 suppression)", syn["suppression_0.8"]["mean_pool_mean"], ci(0, syn["suppression_0.8"]["mean_pool_sd"]), syn["suppression_0.8"]["set_mean"], ci(0, syn["suppression_0.8"]["set_sd"]), "syn"))
panels.append(("Synthetic control,\nadditive (0.0 suppression)", syn["suppression_0.0"]["mean_pool_mean"], ci(0, syn["suppression_0.0"]["mean_pool_sd"]), syn["suppression_0.0"]["set_mean"], ci(0, syn["suppression_0.0"]["set_sd"]), "syn"))
def real(name, r):
    c = r["spearman_ci95_pair_bootstrap"]; s = r["spearman"]
    return (name, s["meanpool_RF"], None, s["set_RF"], None, "real", c["meanpool_RF"], c["set_RF"])
panels.append(real("Real: train Snitz\ntest Ravia (n=195)", d["cross_dataset"]["train_Snitz_test_Ravia"]))
panels.append(real("Real: train Ravia\ntest Snitz (n=360)", d["cross_dataset"]["train_Ravia_test_Snitz"]))
if "spearman" in dj.get("train_Ravia_disjoint_test_Snitz", {}):
    panels.append(real("Real: Ravia, molecule-\ndisjoint, test Snitz", dj["train_Ravia_disjoint_test_Snitz"]))
panels.append(real("Real: Snitz leave-\nmixture-out CV", d["within_dataset_leave_mixture_out"]["Snitz"]))
panels.append(real("Real: Ravia leave-\nmixture-out CV*", d["within_dataset_leave_mixture_out"]["Ravia"]))
fig, ax = plt.subplots(figsize=(11.5, 4.4)); w = 0.36
for i, p in enumerate(panels):
    if p[5] == "syn": lo1 = hi1 = p[2]; lo2 = hi2 = p[4]
    else: lo1, hi1 = p[1] - p[6][0], p[6][1] - p[1]; lo2, hi2 = p[3] - p[7][0], p[7][1] - p[3]
    ax.bar(i - w / 2, p[1], w, yerr=[[lo1], [hi1]], color="#8aa6c1", capsize=3, label="mean-pool (additive)" if i == 0 else None)
    ax.bar(i + w / 2, p[3], w, yerr=[[lo2], [hi2]], color="#b03a6e", capsize=3, label="set-pool (non-additive)" if i == 0 else None)
    ax.text(i, -0.12, f"gap {p[3]-p[1]:+.2f}", ha="center", fontsize=8)
ax.axvline(1.5, color="k", lw=0.8, ls=":")
ax.set_xticks(range(len(panels))); ax.set_xticklabels([p[0] for p in panels], fontsize=7.5); ax.set_ylabel("held-out Spearman"); ax.set_ylim(-0.2, 1.0)
ax.axhline(0, color="k", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("The 0.82 vs 0.25 gap appears only where non-additivity was built in; on leakage-free real human data it is within noise", fontsize=9.5)
fig.text(0.01, 0.005, "Error bars: 95% CI (synthetic: 5 oracle seeds; real: pair bootstrap). *Ravia leave-mixture-out: ~97% of held-out pair molecules also occur in training pairs (mixture_leakage_check.json), so it is not a clean test.", fontsize=6.5)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(os.path.join(FIG, "fig2_mixture_real_vs_synthetic.pdf")); print("ok")
