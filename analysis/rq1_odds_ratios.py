"""RQ1 (Section 4.2): Mantel-Haenszel odds ratios, bootstrap intervals,
observational-unit replication. Reproduces Table 5 and Figure 4."""
import numpy as np, pandas as pd
from common import load_agents, load_events, tables, informative, mh, cluster_boot, mh_parts, results_dir, TOX

ag = load_agents()
keep = [informative(t) for t in tables(ag)]
print(f"agents {len(ag):,} -> informative strata {sum(keep):,}")

# Agents whose system prompt could not be classified -- an API failure, or the
# model echoing its instructions instead of answering -- are excluded from the
# prompted / non-prompted comparison rather than folded into "not prompted".
# Release tables built before this column existed treat every agent as classified.
cls = (ag.prompt_classified.values == 1) if "prompt_classified" in ag else np.ones(len(ag), bool)
print(f"prompt verdict available for {int(cls.sum()):,} agents ({100*cls.mean():.1f}%); "
      f"prompted {int((cls & (ag.prompted.values == 1)).sum()):,}")

rows = []
for name, m in [("All agents", np.ones(len(ag), bool)),
                ("Prompted", cls & (ag.prompted.values == 1)),
                ("Not prompted", cls & (ag.prompted.values == 0))]:
    sel = m & np.array(keep)
    tabs = [t for t, s in zip(tables(ag), sel) if s]
    orr, ci, res = mh(tabs)
    _, ci_ag, k_ag = cluster_boot(tabs, ag.agent[sel], seed=0)
    _, ci_cr, k_cr = cluster_boot(tabs, ag.creator[sel], seed=0)
    rows.append(dict(group=name, OR=round(orr, 3), analytic=np.round(ci, 3).tolist(),
                     agent_boot=np.round(ci_ag, 3).tolist(),
                     creator_boot=np.round(ci_cr, 3).tolist(),
                     strata=len(tabs), responses=int(sum(t.sum() for t in tabs)),
                     agent_clusters=k_ag, creator_clusters=k_cr))
tab5 = pd.DataFrame(rows); print("\n", tab5.to_string(index=False))
tab5.to_csv(f"{results_dir()}/table5_odds_ratios.csv", index=False)

# Absolute rates and the unstratified OR are MARGINAL quantities: computed over
# every agent, not only the informative strata.
a, b, c, d = ag.a_RT_ST.sum(), ag.b_RT_SN.sum(), ag.c_RN_ST.sum(), ag.d_RN_SN.sum()
p_st, p_sn = a/(a+c), b/(b+d)
print(f"\nP(RT|ST)={p_st:.4f}  P(RT|SN)={p_sn:.4f}  marginal RR={p_st/p_sn:.3f}")
print(f"unstratified (marginal) OR = {(a*d)/(b*c):.3f}")
for name, m in [("All agents", np.ones(len(ag), bool)),
                ("Prompted", cls & (ag.prompted.values == 1)),
                ("Not prompted", cls & (ag.prompted.values == 0))]:
    A = ag[m]; aa, bb, cc, dd = A.a_RT_ST.sum(), A.b_RT_SN.sum(), A.c_RN_ST.sum(), A.d_RN_SN.sum()
    print(f"  {name:13s} P(RT|ST)={aa/(aa+cc):.4f}  P(RT|SN)={bb/(bb+dd):.4f}")
print("MH pooled risk ratio (stratified):",
      round(mh([t for t,s in zip(tables(ag), keep) if s])[2].riskratio_pooled, 3))

# joint creator bootstrap for the ratio of odds ratios
mp = cls & (ag.prompted.values == 1) & np.array(keep)
mn = cls & (ag.prompted.values == 0) & np.array(keep)
np_, dp = mh_parts([t for t, s in zip(tables(ag), mp) if s])
nn, dn = mh_parts([t for t, s in zip(tables(ag), mn) if s])
cp, cn = ag.creator[mp].values, ag.creator[mn].values
uniq = np.unique(np.concatenate([cp, cn]))
Mp = {u: np.flatnonzero(cp == u) for u in uniq}; Mn = {u: np.flatnonzero(cn == u) for u in uniq}
rng, out = np.random.default_rng(0), []
for _ in range(2000):
    pick = uniq[rng.integers(0, len(uniq), len(uniq))]
    ip = np.concatenate([Mp[u] for u in pick]); iN = np.concatenate([Mn[u] for u in pick])
    if len(ip) and len(iN):
        out.append((nn[iN].sum()/dn[iN].sum()) / (np_[ip].sum()/dp[ip].sum()))
print("ratio of odds ratios:", np.round(np.percentile(out, [2.5, 50, 97.5]), 3))

# comment-level replication
ev = load_events()
cl = []
for agent, g in ev.groupby("agent", sort=False):
    t = np.array([[((g.resp_toxic == 1) & (g.stim_toxic == 1)).sum(),
                   ((g.resp_toxic == 1) & (g.stim_toxic == 0)).sum()],
                  [((g.resp_toxic == 0) & (g.stim_toxic == 1)).sum(),
                   ((g.resp_toxic == 0) & (g.stim_toxic == 0)).sum()]], float)
    if informative(t): cl.append(t)
orr, ci, _ = mh(cl)
print(f"\ncomment-level OR = {orr:.3f} {np.round(ci,3)} over {len(cl):,} strata")
