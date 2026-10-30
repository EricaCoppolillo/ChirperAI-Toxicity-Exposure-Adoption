"""RQ2 (Section 4.3): discrete-time hazard of a first toxic response. Table 8."""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import load_events, results_dir

ev = load_events()
codes = pd.factorize(ev["agent"])[0]
n = len(ev); first = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
sizes = np.diff(np.r_[first, n]); first_of = np.repeat(first, sizes)
s = ev.stim_toxic.to_numpy(); csum = np.cumsum(s)
ev["cum_prior_toxic"] = csum - np.repeat(csum[first] - s[first], sizes) - s
ev["event_idx"] = np.arange(n) - first_of

ft = ev[ev.resp_toxic == 1].groupby("agent")["event_idx"].min()
ev["_ft"] = ev.agent.map(ft)
surv = ev[ev._ft.isna() | (ev.event_idx <= ev._ft)].copy()
surv["y"] = surv.resp_toxic
cens = surv._ft.isna().groupby(surv.agent).first().sum()
print(f"agent-event observations {len(surv):,}  adoptions {int(surv.y.sum()):,}  censored agents {cens:,}")

surv["log_cum_prior"] = np.log1p(surv.cum_prior_toxic)
surv["log_t"] = np.log1p(surv.event_idx)
surv["month"] = surv.t.dt.tz_localize(None).dt.to_period("M").astype(str)
COLS = ["stim_toxic", "log_cum_prior", "log_t"]

f = surv.reset_index(drop=True)
X = sm.add_constant(pd.concat([f[COLS].astype(float),
                               pd.get_dummies(f["month"], prefix="m", drop_first=True).astype(float)], axis=1))
g = pd.factorize(f.creator.astype(str))[0]
m = sm.Logit(f.y.astype(float), X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": g})
t8 = pd.DataFrame({"hazard_ratio": np.exp(m.params[COLS]), "se": m.bse[COLS], "p": m.pvalues[COLS]})
print("\n", t8)
t8.to_csv(f"{results_dir()}/table8_hazard.csv")

b = m.params["log_cum_prior"]
print(f"\nper doubling: {100*(np.exp(b*np.log(2))-1):+.1f}%   "
      f"10 prior: {np.exp(b*np.log1p(10)):.2f}x   100 prior: {np.exp(b*np.log1p(100)):.2f}x")
