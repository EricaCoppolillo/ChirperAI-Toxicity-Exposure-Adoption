"""Figures 4 and 6 from the released tables (Figure 5 comes from rq2_null_models.py)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from common import load_agents, tables, informative, mh, cluster_boot, results_dir

sns.set_style("whitegrid")
ag, RES = load_agents(), results_dir()
keep = np.array([informative(t) for t in tables(ag)])

# ---- Figure 4: Mantel-Haenszel odds ratios with creator-bootstrap intervals
labels = ["All", "Prompted", "Non-Prompted"]
cls    = (ag.prompt_classified.values == 1) if "prompt_classified" in ag \
         else np.ones(len(ag), bool)
masks  = [np.ones(len(ag), bool),
          cls & (ag.prompted.values == 1),
          cls & (ag.prompted.values == 0)]
means, los, his = [], [], []
for m in masks:
    tabs = [t for t, s in zip(tables(ag), m & keep) if s]
    orr, _, _ = mh(tabs)
    _, ci, _ = cluster_boot(tabs, ag.creator[m & keep], R=2000, seed=0)
    means.append(orr); los.append(ci[0]); his.append(ci[1])
means, los, his = map(np.array, (means, los, his))

colors  = ["gray"] + sns.color_palette()[2:4][::-1]
markers = ["o", "^", "H"]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, c in enumerate(colors):
    ax.errorbar(x[i], means[i], yerr=[[means[i]-los[i]], [his[i]-means[i]]],
                fmt="none", ecolor=c, elinewidth=1.5, capsize=5, capthick=1.5, zorder=2)
    ax.plot(x[i], means[i], lw=0, marker=markers[i], color=c, markersize=12,
            markerfacecolor="white", markeredgewidth=2, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Odds Ratio", fontsize=15)
ax.grid(axis="y", ls="--", alpha=.4); ax.grid(axis="x", alpha=0)
fig.tight_layout(); fig.savefig(f"{RES}/fig4_odds_ratio.pdf", bbox_inches="tight")
fig.savefig(f"{RES}/fig4_odds_ratio.png", dpi=170, bbox_inches="tight")
print("Figure 4:", dict(zip(labels, np.round(means, 3))))

# ---- Figure 6: ITRR / STRR distributions
d = ag.rename(columns={"a_RT_ST": "a", "b_RT_SN": "b",
                       "n_stim_toxic": "nT", "n_stim_nontoxic": "nN"}).copy()
d["ntox"] = d.a + d.b
d = d[~((d.a == 0) & (d.nT == 0)) & ~((d.b == 0) & (d.ntox == 0))]
d["ITRR"], d["STRR"] = d.a / d.nT, d.b / d.ntox
long = pd.DataFrame({"score": pd.concat([d.ITRR, d.STRR]).values,
                     "type":  ["ITRR"] * len(d) + ["STRR"] * len(d)})

fig, ax = plt.subplots(figsize=(5, 4))
pal = [sns.color_palette()[0], sns.color_palette()[1]]
sns.violinplot(data=long, x="type", y="score", hue="type", palette=pal, legend=False,
               inner=None, cut=0, split=False, ax=ax, linewidth=0)
for coll in ax.collections:   # keep only the left half of each violin
    coll.set_alpha(.75)
sns.boxplot(data=long, x="type", y="score", width=.18, showcaps=True, ax=ax,
            boxprops=dict(facecolor="white", zorder=3), showmeans=True,
            meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green"),
            flierprops=dict(marker="o", markersize=2, alpha=.4), zorder=3)
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticks([0, 1]); ax.set_xticklabels(["$\\mathrm{ITRR}$", "$\\mathrm{STRR}$"], fontsize=13)
ax.set_ylim(-.03, 1.03)
fig.tight_layout(); fig.savefig(f"{RES}/fig6_itrr_strr.pdf", bbox_inches="tight")
fig.savefig(f"{RES}/fig6_itrr_strr.png", dpi=170, bbox_inches="tight")
print(f"Figure 6: n={len(d):,}  ITRR mean={d.ITRR.mean():.3f}  STRR mean={d.STRR.mean():.3f}")
