# Harm in AI-Driven Societies: An Audit of Toxicity Adoption on Chirper.ai

> **Paper under submission** — *EPJ Data Science*
> Erica Coppolillo · Luca Luceri · Emilio Ferrara

---

## Overview

This is the official repository for the paper **"Harm in AI-Driven Societies: An Audit of Toxicity Adoption on Chirper.ai"**, currently under review at the *EPJ Data Science* Journal.

The paper investigates the spread and adoption of toxic content within [Chirper.ai](https://chirper.ai), a fully AI-populated social platform. Through a systematic audit, we analyze how toxicity propagates in an environment where all actors are AI agents, offering new insights into the risks and dynamics of AI-driven social ecosystems.

---

## Data Availability

The data used in this study are **proprietary** and cannot be redistributed. If you are interested in accessing the dataset for research purposes, please contact the authors directly.

## What is released

### Derived data products (`data/`)

| File | Rows | Contents |
|---|---|---|
| `agents.csv.gz` | one per chirper | hashed agent and creator ids; `prompted` and `prompt_classified` flags; toxic and non-toxic stimulus counts; the four contingency-table cells `a_RT_ST`, `b_RT_SN`, `c_RN_ST`, `d_RN_SN`; event and post counts |
| `events.csv.gz` | one per stimulus–response pair | hashed event, agent, creator, post and source ids; engagement timestamp; continuous stimulus and response toxicity scores |

`prompted` is 1 when the agent's system prompt was classified as soliciting toxic content.
`prompt_classified` is 1 when a usable verdict exists at all: a small number of prompts
returned an API failure, or a response in which the model echoed its instructions instead of
answering. Those agents are excluded from the prompted / non-prompted comparison in
Section 4.2 rather than folded into the non-prompted group, which would bias it. Filter on
`prompt_classified == 1` before splitting on `prompted`.

`agents.csv.gz` holds the sufficient statistics for every contingency-table analysis in
Sections 4.2 and 4.4: all odds ratios, bootstrap intervals and permutation tests can be
recomputed exactly from those four counts per agent. `events.csv.gz` additionally supports
the event-level models, the survival analysis, the null models, and the threshold-sensitivity
analyses — the continuous scores are retained so that results at cut-offs other than 0.5 can
be reproduced.

### Validation annotations (`data/`)

| File | Contents |
|---|---|
| `annotations_detoxify.csv` | manual toxicity labels against `detoxify` scores, keyed by hashed record id |
| `annotations_gemini.csv` | manual prompt-category labels against `Gemini-2.0-flash` output, keyed by hashed agent id |

Labels only; the annotated text is not released.

### Model configuration (`data/config.json`)

Toxicity and high-toxicity thresholds; `detoxify` model and version; `Gemini-2.0-flash`
version and system prompt reference; `pycld2` configuration; observation window; every
random seed and replicate count used for the bootstrap, permutation and null-model
procedures; the identifier-hashing scheme.

### Intermediate analysis outputs (`data/intermediate/`, `results/`)

Monthly activity and toxicity series (Appendix table); binned adoption curve with null bands
(Figure 5); per-threshold ITRR/STRR correlations and null values (Table 9); fitted
coefficient tables for every model reported.

### Code

| Path | Purpose |
|---|---|
| `make_release.py` | builds `data/` from the raw corpus — run once by the authors |
| `fix_prompted.py` | repairs only the `prompted` / `prompt_classified` columns of an existing `data/`, without re-reading the corpus, and re-runs the Section 4.2 prompted comparison |
| `analysis/common.py` | loaders, Mantel–Haenszel estimator, cluster bootstrap |
| `analysis/rq1_odds_ratios.py` | Section 4.2 |
| `analysis/rq2_null_models.py` | Section 4.3, null models |
| `analysis/rq2_event_level.py` | Section 4.3, event-level model |
| `analysis/rq2_survival.py` | Section 4.3, discrete-time hazard |
| `analysis/rq3_itrr_strr.py` | Section 4.4 |
| `analysis/make_figures.py` | regenerates Figures 1, 4 and 5 from `data/` |
| `ChirperAI-Analysis.ipynb` | the exploratory notebook, for transparency about how the derived tables were produced; requires the raw corpus and is **not** needed to reproduce the paper |


