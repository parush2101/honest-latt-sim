# Honest LATT — simulation prototype

Monte-Carlo prototypes for a research idea in staggered difference-in-differences.

## The idea (one sentence)

HonestDiD keeps the **ATT** as the target and reports a *set*; we instead **change
the estimand** to a credible-subpopulation effect (a "local ATT", LATT) that is
point-identified when the ATT is not. It is not a better estimator — it is a
different estimand choice, defended with selection-aware inference.

The estimator re-weights toward cohorts whose parallel-trends assumption is credible
(based on pre-trend information); the algorithm's data-driven choice is a **selection
event**, and we provide inference valid conditional on it.

## What the simulations show (Tier 1, reduced-form normal model)

- **`tier1.py`** — headline point-estimate demonstration. Under partial parallel-trends
  violation, CS is biased for the ATT it targets (+0.27 here); the reweighted LATT
  recovers the credible-subpopulation effect. Sample-size sweep: LATT bias → 0 as
  precision grows, while CS bias is frozen (wrong-target vs noise).

- **`master_axis.py` / `master_axis.png`** — the scope-condition figure. Everything is
  governed by **pre-trend informativeness**. CS bias is flat regardless of it; the LATT's
  advantage is the gap between the lines — zero when pre-trends are uninformative, maximal
  when informative. Causal-coverage collapse at low info is driven by the **identification
  gap** (dirty cohorts surviving selection), NOT by selection distortion — sample-splitting,
  which handles selection perfectly, collapses identically.

- **`layer2.py` / `layer2.png`** — RR-style Δ bounds on the selected set. Key finding:
  a **relative-magnitudes** bound (anchored to the observed pre-trends) *inherits* the
  informativeness scope and gives tight-but-wrong intervals when pre-trends are flat
  (false precision). An **absolute / theory-anchored** bound escapes the scope condition
  at the cost of requiring external knowledge. Through-line: when pre-trends can't carry
  the credibility, the economics has to.

## Priority ordering of contributions (as the sims revealed)

1. Estimand choice + reweighted LATT point estimate — the spine.
2. Δ-robustness on the selected set (Layer 2) — larger practical payoff (identification
   gap dominates).
3. Selective inference (Layer 1) — a genuine but second-order refinement.

## Status

All results are **Tier 1** (reduced-form normal model), the clean-lab setting where the
underlying theory is exact. Tier 2 (full panel micro-simulation with the actual CS
estimator) is not yet built.

## Reference papers

- Roth, Sant'Anna, Bilinski & Poe (2023), *What's trending in difference-in-differences?*
- Rambachan & Roth (2023), *A More Credible Approach to Parallel Trends* (HonestDiD).
- Callaway & Sant'Anna (2021), *Difference-in-Differences with multiple time periods*.
