# 🚀 Target Label: Capitalized Net Occupant Wealth ($W_i$)

Reference for the occupant-wealth indices built in
[`process_acs.py`](process_acs.py) (`wealth_from_inputs`). This documents what the
**code actually computes**, the hyper-parameters, and the arbitrary modelling choices.

> ⚠️ **Divergence from the original proposal.** The first design (and the issue
> template) used per-capita *income* (B19301) capitalized as a $1/r$ perpetuity, the
> homeownership rate $\alpha_i$, and *median* home value (B25077). The implemented code
> instead uses per-capita *labor earnings* (B20003) over a **finite annuity**, a
> **per-capita** equity term, and **mean** home value. The reasons are noted inline
> below. If the paper still describes the perpetuity/median form, that is drift to fix.

## 📌 Motivation (The "Why")

The model targets the *permanent wealth* capitalized into the built environment, not a
single-period flow. Income alone distorts ranks via (1) the **renter/owner asymmetry**
(equal incomes ≠ equal wealth) and (2) **leverage** (ignoring mortgage debt overstates
the wealth of highly-leveraged gentrifying tracts vs. high-equity established ones).

The fix is a tract-level **Net Occupant Wealth Index $W_i$** = present value of human
capital + net housing equity. Three nested variants ($W_1 \subset W_2 \subset W_3$)
trade simplicity for realism; $W_1$ is the frozen baseline, $W_2$/$W_3$ are extensions.

## 🧮 The Three Formulas

All three share the same per-capita human-capital and equity skeleton and are computed
**once per discount rate** $r \in$ `DISCOUNT_RATES`. Let $\text{pop}_i$ = total
population (B01001_001E).

**Human capital (all variants):** per-capita *labor earnings* (B20003 aggregate
earnings $/\text{pop}$) capitalized over a **finite** horizon as an ordinary annuity:

```math
\text{HC}_i(r, N) = \frac{\text{earnings}_i}{\text{pop}_i}\times a(r, N),
\qquad a(r,N) = \frac{1 - (1+r)^{-N}}{r}
```

> *Why earnings, not income, and finite, not perpetuity:* B20003 (wages +
> self-employment) excludes interest/dividends/rent, so capitalizing it does **not**
> double-count the capital-income and housing-equity terms. A finite $N$-year annuity
> reflects that labor income stops at retirement; $a(r,N)\to 1/r$ as $N\to\infty$.

**Net housing equity (all variants):** per-capita aggregate owner value (B25082)
de-levered by the mortgaged share $m_i = $ B25081_002E / B25081_001E:

```math
\text{EQ}_i(\text{LTV}) = \frac{\text{agg\_owner\_value}_i}{\text{pop}_i}\times
\big(\underbrace{1 - m_i\cdot\text{LTV}}_{\text{freeclear}\cdot 1 + \text{mortgaged}\cdot(1-\text{LTV})}\big)
```

| Variant | Formula | Horizon $N$ | LTV | Capital income |
| :--- | :--- | :--- | :--- | :--- |
| **$W_1$** (baseline) | $\text{HC}(r, 20) + \text{EQ}(\text{LTV}_{macro})$ | fixed `20` | flat `0.50` | — |
| **$W_2$** | $\text{HC}(r, N_i) + \text{CAP}(r_k) + \text{EQ}(\text{LTV}_2)$ | tract $N_i$ | Fed year-matched | ✓ |
| **$W_3$** | $\text{HC}(r, N_i) + \text{CAP}(r_k) + \text{EQ}(\text{LTV}_{3,i})$ | tract $N_i$ | imputed per tract | ✓ |

**$W_2$/$W_3$ capital-income term** — per-capita interest+dividend+net-rent (B19064)
as a **perpetuity** at gross asset yield $r_k$ (financial assets don't expire at
retirement, hence $1/r_k$ not a finite annuity):

```math
\text{CAP}_i(r_k) = \frac{1}{r_k}\times\frac{\text{agg\_capital\_income}_i}{\text{pop}_i}
```

- **$N_i$ (W2/W3 horizon):** `expected_working_years` — population-weighted
  $\sum_a s_{a,i}\max(0,\, 65 - \bar a)$ over the 16–64 age brackets (B01001). Retiree-heavy
  tracts get a short annuity; young tracts approach $65-16$.
- **$\text{LTV}_2$ (W2):** $1 - $ Fed owners'-equity share (FRED **HOEREPHRE**),
  averaged over the ACS 5-year window $Y\!-\!4..Y$. A sourced, time-varying replacement
  for the flat $\text{LTV}_{macro}$ guess.
- **$\text{LTV}_{3,i}$ (W3):** $(\text{ORIG\_LTV}\times\text{amort}(\tau_i))/\text{appreciation}_i$,
  clipped to $[0,1]$. Tenure $\tau_i$ from the B25038 owner move-in distribution;
  amortization from a 30-yr / 5% schedule; appreciation from FHFA HPI (county-first,
  CBSA-fallback) since the median move-in year. Long-tenure / appreciated tracts → low
  LTV (high equity).

**Diagnostic level $V_i$** (not an index; not z-scored) — **mean** owner-occupied home
value $=$ B25082 aggregate value / B25003_002E owner-occupied units.

## 📊 Parameters & Hyper-parameters

| Symbol | Code constant | Value | Meaning / source |
| :--- | :--- | :--- | :--- |
| $r$ | `DISCOUNT_RATES` | `(0.02, 0.03, 0.05, 0.07)` | Net-of-growth labor discount rate; one $W$ column per rate |
| $N$ ($W_1$) | `HUMAN_CAPITAL_YEARS` | `20` | Fixed finite annuity horizon (baseline only) |
| $\text{LTV}_{macro}$ | `LTV_MACRO` | `0.50` | Flat mortgaged LTV ($W_1$); SCF/AHS ballpark |
| $r_k$ | `CAPITAL_YIELD` | `0.045` | Gross asset yield for the $W_2$/$W_3$ capital term |
| $r_k$ grid | `CAPITAL_YIELD_GRID` | `(0.040, 0.045, 0.050)` | $r_k$ sensitivity report only |
| exit age | `WORKING_EXIT_AGE` | `65` | Age the labor flow stops (drives $N_i$) |
| $\text{ORIG\_LTV}$ | `ORIG_LTV` | `0.80` | Purchase-origination LTV ($W_3$) |
| mortgage rate | `MORTGAGE_RATE` | `0.05` | Amortization schedule rate ($W_3$) |
| mortgage term | `MORTGAGE_TERM` | `30` | Amortization schedule term, years ($W_3$) |
| equity window | `OWNERS_EQUITY_WINDOW` | `5` | Years averaged for $\text{LTV}_2$ (matches ACS 5-yr) |
| replicates | `N_REPLICATES` | `80` | Census VRE successive-difference replicates |
| SDR scale | `SDR_SCALE` | `4/80` | $\text{Var}(X)=\tfrac{4}{80}\sum_r (X_r-X_0)^2$ |

### ACS source tables

| Input | Table | Code column |
| :--- | :--- | :--- |
| Labor earnings (aggregate) | B20003_001E | `aggregate_earnings_usd` |
| Owner value (aggregate) | B25082_001E | `aggregate_owner_value_usd` |
| Capital income (aggregate) | B19064_001E | `aggregate_capital_income_usd` |
| Population | B01001_001E | `total_population` |
| Tenure (owner-occupied) | B25003_001E / _002E | occupied total / owner |
| Mortgage status | B25081_001E / _002E | owners / with-mortgage |
| Age distribution → $N_i$ | B01001 (16–64 brackets) | `WORKING_AGE_VARS` |
| Owner tenure → $W_3$ LTV | B25038_003E…008E | `MOVEIN_OWNER_VARS` |
| Per-capita income (label $Y$) | B19301 | `per_capita_income_usd` |

### External sources

- **FRED HOEREPHRE** (owners' equity as % of household real estate) → $\text{LTV}_2$.
  Fallback table used if CSV absent (approximate).
- **FHFA HPI** (all-transactions, annual; county + CBSA) → $W_3$ appreciation. Absent →
  appreciation = 1.0 (no revaluation), $W_3$ still computes.

## 💡 Methodological notes & arbitrary choices

- **Per-capita everything.** HC and EQ both divide by total population (not households /
  owner units), so the index is directly comparable across tracts with different tenure
  mixes. This replaces the proposal's $\alpha_i \times V_i$ ownership-rate weighting —
  the mathematics already zeroes renters' equity contribution via $m_i$/aggregate value.
- **No mortgage-flow deduction.** Principal repayment converts cash → illiquid equity; it
  is not consumption, so income is never reduced before capitalizing.
- **Relative scores, not levels.** Only `Rel_Score_{var}` (per-CBSA, per-year z-score of
  $\log W$) feeds the model and the structural-change test — so city-wide nominal drift
  (inflation, aggregate housing appreciation) cancels each year.
- **Uncertainty.** $W$ relative-score SEs come from the **VRE replicate variance** (SDR,
  80 replicates), capturing cross-table covariance the delta method misses. Available
  from 2014 only, so the wealth structural-change test starts at 2014 (income at 2011).
- **Robustness.** $W_2$/$W_3$ vary $r$ and $r_k$ over their grids; the Spearman-vs-income
  and $r_k$-sensitivity CSVs confirm ranks barely move with the rate choices.
