"""Generate the main paper figure: chemical-class pleasantness lifts (real Keller 2016 data)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from real_data import build_keller_pleasantness
import composition as comp

df = build_keller_pleasantness()
stats = comp.class_pleasantness_stats(df)
by = stats["by_class"]
items = sorted(by.items(), key=lambda kv: kv[1]["lift_vs_overall"])
labels = [k.replace("_", " ") for k, _ in items]
lifts = [v["lift_vs_overall"] for _, v in items]
ns = [v["n"] for _, v in items]
colors = ["#c0504d" if x < 0 else "#4f81bd" for x in lifts]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
y = np.arange(len(labels))
ax.barh(y, lifts, color=colors, edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
for i, (x, n) in enumerate(zip(lifts, ns)):
    ax.text(x + (0.4 if x >= 0 else -0.4), i, f"n={n}", va="center",
            ha="left" if x >= 0 else "right", fontsize=7, color="#444")
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("Mean pleasantness lift vs. overall (0-100 human panel scale)", fontsize=9)
ax.set_title("Which chemical classes make a molecule pleasant\n(real data, Keller & Vosshall 2016; n per class shown)",
             fontsize=10)
ax.margins(y=0.01)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("class_pleasantness.pdf", bbox_inches="tight")
plt.savefig("class_pleasantness.png", dpi=150, bbox_inches="tight")
print("saved figure; classes:", len(labels))
