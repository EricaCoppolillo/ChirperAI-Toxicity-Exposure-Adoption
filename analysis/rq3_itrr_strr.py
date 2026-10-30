"""RQ3 (Section 4.4): ITRR / STRR and their permutation null. Table 9, Figure 6."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from common import load_agents, results_dir

ag = load_agents()
d = ag.rename(columns={"a_RT_ST": "a", "b_RT_SN": "b",
                       "n_stim_toxic": "nT", "n_stim_nontoxic": "nN"}).copy()
d["ntox"] = d.a + d.b
keep = ~((d.a == 0) & (d.nT == 0)) & ~((d.b == 0) & (d.ntox == 0))
print(f"chirpers {len(d):,} | no toxic stimulus {(d.nT==0).sum():,} | "
      f"no toxic response {(d.ntox==0).sum():,} | retained {keep.sum():,}")

d = d[keep].copy()
d["ITRR"], d["STRR"] = d.a / d.nT, d.b / d.ntox
print(f"ITRR at 0: {100*(d.ITRR==0).mean():.1f}%   STRR at 1: {100*(d.STRR==1).mean():.1f}%")

def perm_null(s, R=1000, seed=0):
    rng = np.random.default_rng(seed)
    nT, nN = s.nT.to_numpy(), s.nN.to_numpy()
    tot = np.minimum(s.ntox.to_numpy(), nT + nN)
    out = np.empty(R)
    for r in range(R):
        a0 = rng.hypergeometric(np.maximum(nT, 0), np.maximum(nN, 0), tot)
        m = (nT > 0) & (tot > 0)
        out[r] = spearmanr(a0[m] / nT[m], (tot - a0)[m] / tot[m])[0]
    return out

rows = []
for m in [1, 2, 3, 5, 10, 20]:
    s = d[d.ntox >= m]
    if len(s) < 50: continue
    nul = perm_null(s, R=500)
    rows.append(dict(m=m, agents=len(s),
                     observed=round(spearmanr(s.ITRR, s.STRR)[0], 3),
                     null_median=round(float(np.median(nul)), 3),
                     null_lo=round(float(np.percentile(nul, 2.5)), 3),
                     null_hi=round(float(np.percentile(nul, 97.5)), 3)))
t9 = pd.DataFrame(rows); print("\n", t9.to_string(index=False))
t9.to_csv(f"{results_dir()}/table9_itrr_strr_null.csv", index=False)
print(f"\nfull sample: pearson {pearsonr(d.ITRR, d.STRR)[0]:+.4f}  "
      f"spearman {spearmanr(d.ITRR, d.STRR)[0]:+.4f}  n={len(d):,}")

e = ag[(ag.n_stim_toxic > 0) & (ag.n_stim_nontoxic > 0) &
       (ag.a_RT_ST + ag.b_RT_SN > 0)]
x, y = e.a_RT_ST / e.n_stim_toxic, e.b_RT_SN / e.n_stim_nontoxic
print(f"disjoint variant (n={len(e):,}): pearson {pearsonr(x,y)[0]:+.4f}  "
      f"spearman {spearmanr(x,y)[0]:+.4f}")
