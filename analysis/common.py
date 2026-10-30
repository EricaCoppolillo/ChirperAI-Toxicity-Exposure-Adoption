"""Shared loaders and estimators. All analyses read only release/data/."""
import json, os
import numpy as np, pandas as pd
import statsmodels.stats.contingency_tables as ct

DATA = os.environ.get("CHIRPER_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
CFG  = json.load(open(os.path.join(DATA, "config.json")))
TOX  = CFG["toxicity_threshold"]

def load_agents():
    return pd.read_csv(os.path.join(DATA, "agents.csv.gz"))

def load_events():
    ev = pd.read_csv(os.path.join(DATA, "events.csv.gz"))
    ev["t"] = pd.to_datetime(ev.t_ms, unit="ms", utc=True)
    ev["stim_toxic"] = (ev.stim_tox >= TOX).astype(int)
    ev["resp_toxic"] = (ev.resp_tox >= TOX).astype(int)
    return ev.sort_values(["agent", "t_ms"]).reset_index(drop=True)

def tables(ag):
    """Per-agent 2x2 tables [[a,b],[c,d]] in the order of ag."""
    return [np.array([[r.a_RT_ST, r.b_RT_SN], [r.c_RN_ST, r.d_RN_SN]], float)
            for r in ag.itertuples()]

def informative(t):
    """Agent exposed to both stimulus types AND varying in response toxicity."""
    return t.sum() > 1 and (t.sum(1) > 0).all() and (t.sum(0) > 0).all()

def mh(tabs):
    r = ct.StratifiedTable(tabs)
    return r.oddsratio_pooled, r.oddsratio_pooled_confint(), r

def mh_parts(tabs):
    A = np.stack(tabs); n = A.sum(axis=(1, 2))
    return A[:, 0, 0] * A[:, 1, 1] / n, A[:, 0, 1] * A[:, 1, 0] / n

def cluster_boot(tabs, clusters, R=2000, seed=0):
    """Resample whole clusters; recompute the pooled MH odds ratio."""
    num, den = mh_parts(tabs)
    cl = np.asarray(clusters); uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
    rng = np.random.default_rng(seed); out = np.empty(R)
    for i in range(R):
        sel = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        out[i] = num[sel].sum() / den[sel].sum()
    return num.sum() / den.sum(), np.percentile(out, [2.5, 97.5]), len(uniq)

def results_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d
