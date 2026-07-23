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

- **`tier2.py` / `tier2_figure.py` / `tier2.png`** — full staggered PANEL micro-simulation.
  Generates individual panel data, estimates cohort-level pre/post coefficients with real
  DiD contrasts vs a shared never-treated control, and estimates the covariance Sigma_hat
  FROM THE DATA. The spine survives: CS bias flat at +0.30, LATT bias falls to ~0, same
  scope pattern as Tier 1. Sigma_hat is verified calibrated where selection is deterministic
  (coverage 94-95% at high info); the SE gap at moderate info (est 0.079 vs MC 0.119) is the
  selection-induced variance the conditional SE ignores — independent validation of why
  Layer 1 / sample-splitting is needed.

- **`layer2_full.py` / `layer2_full_figure.py` / `layer2_full.png`** — the full Layer 2
  machinery (load-bearing contribution). HonestDiD-style FLCI for the smoothness class
  SD(M): a linear-extrapolation estimator + the max-bias LP + the folded-normal critical
  value cv_{1-alpha}(bbar/sigma). VALIDATED by coverage simulation (covers iff M >= true
  curvature; both point estimates collapse to 0% once any curvature appears). Headline:
  the FLCI restores honest coverage where the point estimate collapses, pays with width
  set by the assumed M, and yields an interpretable breakdown value. Because SD keys off
  smoothness STRUCTURE (not pre-trend magnitude), it escapes the relative-magnitudes
  informativeness trap documented in the Layer 2 prototype. Reduced-form normal model
  (known Sigma); integration with multi-cohort selection/aggregation and estimated Sigma
  is the next step.

## Priority ordering of contributions (as the sims revealed)

1. Estimand choice + reweighted LATT point estimate — the spine.
2. Δ-robustness on the selected set (Layer 2) — larger practical payoff (identification
   gap dominates).
3. Selective inference (Layer 1) — a genuine but second-order refinement.

## Status

**Tier 1** (reduced-form normal model) and **Tier 2** (full panel micro-simulation with
real DiD estimation and an estimated covariance) both confirm the spine. Still open:
the selective (truncated-normal) CI vs sample-splitting length race (Layer 1 keep/cut
decision), and the full ARP/FLCI Layer 2 machinery.

## Reference papers

- Roth, Sant'Anna, Bilinski & Poe (2023), *What's trending in difference-in-differences?*
- Rambachan & Roth (2023), *A More Credible Approach to Parallel Trends* (HonestDiD).
- Callaway & Sant'Anna (2021), *Difference-in-Differences with multiple time periods*.

## Layer 1 keep/cut race (inconclusive on selective, decisive on the decision)

`race.py` / `race2.py` / `race3.py`. Findings:
- Selection distortion is small: naive coverage stays ~94% even in an adversarial
  near-threshold, identification-free design (confirmed across all tiers).
- Sample-splitting restores ~95% coverage at a ~30-55% interval-length premium.
- CAVEAT: the exact polyhedral (Lee-Sun-Sun-Taylor) selective CI here is NOT reliable
  — it fails the single-cohort sanity check (63.9% coverage, should be ~95%), so its
  length numbers are not trustworthy. Separately, ~10-23% of aggregate selective
  intervals blow up to infinite length (the known non-randomized pathology, not a bug).

Provisional verdict: CUT Layer 1, default to sample-splitting. Resurrect selective
inference only in its hybrid/randomized (data-carving) form, and only if target
applications are small-sample enough that splitting's ~50% width premium bites.

## Focused core: does selection distort the Layer 2 FLCI? (selection_flci_core.py)

The one genuinely untested question: applying HonestDiD to a data-DEPENDENT selected set.
full-data (select+FLCI on same draw) vs split (select on half 1, FLCI on half 2).

Findings (reduced-form normal, tau_3 target, strong pre/post correlation):
- NO systematic full-vs-split coverage gap (<=3pp, inconsistent sign). The feared
  selection-sampling distortion of the FLCI is not supported, and splitting is NOT the
  remedy. Selection distortion on the estimator center is small (0.02-0.09) and, where
  present, makes full-data LESS biased than split -- so splitting would slightly hurt.
  => The integrated procedure is valid WITHOUT splitting (simpler than feared).
- REAL issue surfaced instead: because the selected set is random, its residual violation
  (curvature) is random, so a fixed smoothness bound M must be set to the WORST-CASE
  selected composition (max, not mean, residual curvature) to guarantee coverage. A mean-
  calibrated M undercovers -- for BOTH full-data and split. Remedy: conservative M, or a
  selection rule that caps per-cohort curvature. Not fixed by splitting.

## Fine M-sweep: calibrating M to the random selected set (m_sweep.py)

Refines the composition-randomness finding. Coverage vs M, full-data vs split, with the
residual-curvature distribution of the (random) selected set:
- Mean-calibrated M UNDERCOVERS: mean residual curvature 0.031 -> ~88% coverage.
- 95% coverage reached at M~0.039 -- ABOVE the mean, but BELOW the strict max (0.050):
  the FLCI's sampling slack buys some buffer, so M need not reach the worst case, but it
  must clearly exceed the mean.
- full-data and split cross 95% at the SAME M (0.039) -> splitting is not the remedy;
  the calibration issue is about M vs the random selected composition, not selection
  sampling distortion. Confirms: integrated procedure valid without splitting, but M must
  be set above the mean residual curvature of the selected set.

## Empirical application: Community Health Centers (chc_figure.py)

The paper's empirical application uses the county-level staggered rollout of Community
Health Centers from Bailey & Goodman-Bacon (2015), "The War on Poverty's Experiment in
Public Medicine," *American Economic Review* 105(3). The method flags three suspect
adoption cohorts (1965, 1970, 1973) with large differential pre-trends and shows that the
credible-subpopulation LATT recovers a less-attenuated elderly-mortality effect (~-2%)
than the pooled ATT; the correction is largest under equal weighting (the suspect cohorts
are small-population counties). `chc_figure.py` reproduces the diagnostic figure.

DATA NOTE: the raw data is not redistributed here (it carries the openICPSR license).
Download the replication package from openICPSR project 112871
(https://doi.org/10.3886/E112871V1); the script reads a slim county-year extract
(fips, year, chc_year_exp, copop, amr_eld) built from its aer_data.dta.
