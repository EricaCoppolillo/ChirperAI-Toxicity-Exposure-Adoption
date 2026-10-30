"""RQ2 (Section 4.3): adoption curve against two null models. Figure 5."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from common import load_events, results_dir, TOX

ev = load_events()
n_ser = ev.groupby("agent")["post"].nunique()
k_ser = (ev[ev.resp_toxic == 1].groupby("agent")["post"].nunique()
           .reindex(n_ser.index, fill_value=0))
assert (k_ser <= n_ser).all()
n_arr, adopt = n_ser.to_numpy(), (k_ser > 0).to_numpy()
rates, p_hat = (k_ser / n_ser).to_numpy(), k_ser.sum() / n_ser.sum()

N_BINS, R, MIN_AGENTS = 20, 500, 100
edges = np.unique(np.round(np.logspace(0, np.log10(n_arr.max()), N_BINS)).astype(int))
edges = np.r_[edges[0] - 1, edges]
bid = np.digitize(n_arr, edges[1:], right=True)
lo, hi = edges[:-1] + 1, edges[1:]
centre = np.sqrt(lo * hi)

obs_p = np.array([adopt[bid == b].mean() if (bid == b).any() else np.nan for b in range(len(centre))])
obs_n = np.array([(bid == b).sum() for b in range(len(centre))])

rng = np.random.default_rng(0)
sim = np.full((R, len(centre)), np.nan)
for r in range(R):
    a = rng.binomial(n_arr, rng.choice(rates, size=len(n_arr))) > 0
    for b in range(len(centre)):
        m = bid == b
        if m.any(): sim[r, b] = a[m].mean()
null_lo, null_md, null_hi = np.nanpercentile(sim, [2.5, 50, 97.5], axis=0)
homog = 1 - (1 - p_hat) ** centre

tbl = pd.DataFrame({"n_lo": lo, "n_hi": hi, "agents": obs_n, "observed": obs_p,
                    "null_lo": null_lo, "null_md": null_md, "null_hi": null_hi})
tbl["excess"] = tbl.observed - tbl.null_md
tbl.to_csv(f"{results_dir()}/fig5_adoption_vs_null.csv", index=False)
m = tbl.agents >= MIN_AGENTS
print(tbl.round(3).to_string(index=False))
print(f"\np_hat={p_hat:.4f}  agent-weighted excess="
      f"{np.average(tbl.excess[m], weights=tbl.agents[m]):+.4f}  "
      f"bins above null 97.5pct: {int((tbl.observed[m] > tbl.null_hi[m]).sum())}/{int(m.sum())}")

fig, ax = plt.subplots(figsize=(4.6, 3.1))
ax.fill_between(centre, null_lo, null_hi, color="#9ecae1", alpha=.55, lw=0, zorder=1,
                label="null: activity and toxicity rate preserved (95%)")
ax.plot(centre, null_md, color="#3182bd", lw=1, zorder=2)
ax.plot(centre, homog, color="#3182bd", ls=":", lw=1.2, zorder=2, label=r"null: $1-(1-p)^{n_S}$")
solid = obs_n >= MIN_AGENTS
ax.plot(centre, obs_p, color="#d95f02", lw=1.6, zorder=4, label="observed")
ax.scatter(centre[solid], obs_p[solid], s=18, color="#d95f02", edgecolor="white", lw=.6, zorder=5)
ax.scatter(centre[~solid], obs_p[~solid], s=16, facecolor="white", edgecolor="#d95f02", lw=1, zorder=5)
ax.set_xscale("log"); ax.set_xlabel(r"$n_\mathcal{S}$"); ax.set_ylabel(r"P$(\mathcal{R}_T\mid n_\mathcal{S})$")
ax.set_ylim(-.02, 1.03); ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(frameon=False, fontsize=7, loc="upper left"); fig.tight_layout()
fig.savefig(f"{results_dir()}/fig5_adoption_vs_null.pdf", bbox_inches="tight")
print("wrote fig5_adoption_vs_null.pdf")
