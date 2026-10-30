"""RQ2 (Section 4.3): event-level model and permutation null. Tables 6-7.

Memory-lean: the design matrix is built once as a single float64 array and
reused for every fit, instead of being rebuilt by pandas for each clustering.
"""
import gc
import numpy as np, pandas as pd, statsmodels.api as sm
from common import load_events, results_dir

ev = load_events()
codes = pd.factorize(ev["agent"])[0]
s  = ev.stim_toxic.to_numpy()
t_ = ev.t_ms.to_numpy().astype("int64")
n  = len(ev)

first   = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
sizes   = np.diff(np.r_[first, n])
first_of = np.repeat(first, sizes)
csum    = np.cumsum(s)
cum_prior = csum - np.repeat(csum[first] - s[first], sizes) - s
event_idx = np.arange(n) - first_of

W   = 24 * 3_600_000
OFF = (t_.max() - t_.min()) + W + 1
tg  = t_ + codes.astype("int64") * OFF
left = np.maximum(np.searchsorted(tg, tg - W, side="left"), first_of)
recent = cum_prior - cum_prior[left]
assert (recent >= 0).all() and (recent <= cum_prior).all()
del tg, left, csum, t_; gc.collect()

month = ev.t.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy()
agent, post, creator = ev.agent.to_numpy(), ev.post.to_numpy(), ev.creator.to_numpy()
y_all = ev.resp_toxic.to_numpy().astype(float)
del ev; gc.collect()

# agents whose responses vary in toxicity
v = pd.DataFrame({"a": codes, "y": y_all}).groupby("a")["y"].transform("nunique").to_numpy() > 1
print(f"agents contributing: {len(np.unique(codes[v])):,}  events: {int(v.sum()):,}", flush=True)

def sel(x): return x[v]
stim, lcum = sel(s).astype(float), np.log1p(sel(cum_prior))
lrec, lev_ = np.log1p(sel(recent)), np.log1p(sel(event_idx))
y = sel(y_all)
ag_, po_, cr_, mo_ = sel(agent), sel(post), sel(creator), sel(month)
del s, cum_prior, recent, event_idx, y_all, agent, post, creator, month; gc.collect()

COLS = ["stim_toxic", "log_cum_prior", "log_recent", "stim_x_cum",
        "log_prior_events", "stim_x_events"]
base = np.column_stack([stim, lcum, lrec, stim * lcum, lev_, stim * lev_])
del lrec; gc.collect()

# Mundlak means (within-agent identification) + month fixed effects
acode = pd.factorize(ag_)[0]
bars = np.empty_like(base)
cnt = np.bincount(acode)
for j in range(base.shape[1]):
    bars[:, j] = (np.bincount(acode, weights=base[:, j]) / cnt)[acode]
D = pd.get_dummies(pd.Series(mo_), prefix="m", drop_first=True).to_numpy(dtype=float)
X = np.column_stack([np.ones(len(y)), base, bars, D])
names = ["const"] + COLS + [c + "_bar" for c in COLS] + [f"m{i}" for i in range(D.shape[1])]
del bars, D; gc.collect()
print(f"design matrix: {X.shape}  ({X.nbytes/1e9:.2f} GB)", flush=True)

idx = {c: names.index(c) for c in COLS}
def report(res):
    p, se, pv = res.params, res.bse, res.pvalues
    return pd.DataFrame({"OR": [np.exp(p[idx[c]]) for c in COLS],
                         "se": [se[idx[c]] for c in COLS],
                         "p":  [pv[idx[c]] for c in COLS]}, index=COLS)

for lab, g in [("agent", acode), ("post", pd.factorize(po_)[0]),
               ("creator", pd.factorize(cr_)[0])]:
    r = sm.Logit(y, X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": g})
    out = report(r)
    print(f"\n=== clustered on {lab} ({len(np.unique(g)):,} clusters) ===\n{out}", flush=True)
    if lab == "creator":
        out.to_csv(f"{results_dir()}/table6_event_level.csv")
    del r; gc.collect()

# Table 7: attenuation by toxic share of exposure history
fit0 = sm.Logit(y, X).fit(disp=False)
p0, E = fit0.params, np.log1p(200)
rows = [(sh, round(float(np.exp(p0[idx["stim_toxic"]]
                               + p0[idx["stim_x_cum"]] * np.log1p(200 * sh)
                               + p0[idx["stim_x_events"]] * E)), 2))
        for sh in (.01, .05, .20, .50)]
t7 = pd.DataFrame(rows, columns=["toxic_share", "OR_toxic_stimulus"])
print("\n", t7.to_string(index=False), flush=True)
t7.to_csv(f"{results_dir()}/table7_attenuation.csv", index=False)

# Permutation null: shuffle response toxicity within agent
R = 50
st = np.flatnonzero(np.r_[True, acode[1:] != acode[:-1]])
bounds = list(zip(st, np.r_[st[1:], len(y)]))
rng = np.random.default_rng(0)
obs = np.array([p0[idx[c]] for c in COLS])
null = np.empty((R, len(COLS)))
for i in range(R):
    yp = y.copy()
    for lo, hi in bounds:
        yp[lo:hi] = rng.permutation(yp[lo:hi])
    pp = sm.Logit(yp, X).fit(disp=False).params
    null[i] = [pp[idx[c]] for c in COLS]
    if (i + 1) % 10 == 0: print(f"  permutation {i+1}/{R}", flush=True)
out = pd.DataFrame({"observed": obs, "null_mean": null.mean(0), "null_sd": null.std(0)}, index=COLS)
out["z_vs_null"] = (out.observed - out.null_mean) / out.null_sd
print("\n--- permutation null ---\n", out)
out.to_csv(f"{results_dir()}/permutation_null.csv")
