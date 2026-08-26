# Carved procedure on the shale application (Refine #5, empirical half)

Script: `housing/carved_application.py` (run from repo root). Reuses the same
USDA-onset + FHFA-HPI cohort event studies as `fracking_figure.py`, estimates the
covariance of the stacked cohort x event-time coefficient vector by cluster
bootstrap (B=800; the estimated-Sigma case, Remark 3), and inverts the randomized
truncated-normal pivot of Theorem 1 for the size-weighted credible-subpopulation
LATT.

## What it produces (the pieces Refine #5 named)

- **Estimated covariance** Sigma-hat (72x72, bootstrap) — the "augmented covariance."
- **Conditioned selection cell** at c=0.03: retained {2003, 2004, 2005}, dropped
  {2006..2011}, with each cohort's distance to the screen in bootstrap-SE units.
- **Randomization scale** gamma, and the **carved endpoints** for the LATT.

## Key numbers (size-weighted LATT, matching Definition 1 and fracking_figure.py)

Credible LATT point = **-0.002 log pts** (t = -0.12) — matches the paper exactly.

| interval | 95% set (log pts) | width | what it carries |
|---|---|---|---|
| naive z | [-0.035, +0.031] | 0.065 | nothing (ignores selection) |
| carved, gamma=0 | [-0.043, +0.033] | 0.076 | selection uncertainty (Theorem 1) |
| carved, gamma=0.5 | [-0.081, +0.045] | 0.125 | selection, more randomized |
| **Corollary 1 (carve + M=c)** | **[-0.073, +0.063]** | 0.136 | **selection + identification (data-driven honest)** |
| near-optimal FLCI at M=c | [-0.059, +0.055] | 0.115 | identification only (valid under separation) |

The near-optimal FLCI **[-0.059, +0.055] reproduces the paper's [-5.9, +5.5] log
points exactly.** All constructions put the credible effect indistinguishable from
zero; the fully honest data-driven object (Corollary 1) still contains 0 and its
upper edge (+0.063) sits below the pooled ATT (+0.067).

## The honest reading (correcting an earlier over-simplification)

This is NOT a clean separation case. **4 of 9 cohorts sit within ~1 bootstrap SE of
the screen** (2004, 2005 borderline-in; 2006, 2009 borderline-out); only 2003 is
firmly retained and 2007/2008/2011 firmly dropped. So data-driven selection genuinely
injects uncertainty, and the carved procedure is exactly what carries it. The result
is that the **near-zero credible reading is robust to that selection uncertainty**:
the LATT stays ~0 across the plausible selected sets, the carved (selection) interval
at gamma=0 tracks the naive one, and composing with the level bound gives the honest
data-driven set without disturbing the conclusion.

This discharges the empirical half of Refine #5: the carved procedure now runs on the
real data with estimated Sigma, exhibiting gamma, the selection cell, and the carved
endpoints, rather than only a fixed-set sensitivity interval after same-data selection.

## Step 5 of Algorithm 1 (the trade): D-hat and the composition-gap breakdown Gamma*

- ATT (all 9 cohorts)   = +0.0672 (SE 0.0081)
- LATT (credible 3)     = -0.0020 (SE 0.0166)
- D = ATT - LATT        = +0.0692 log pts (SE 0.0129, t = +5.36)  [matches paper's +6.9]
- Var_ATT = 0.00007, Var_LATT = 0.00028, DeltaVar = +0.00021 (dropping cohorts raises variance)
- Gamma* = (DeltaVar - D^2)/(2D) = **-0.0331 log pts (-3.3 log points)**

Interpretation: D = B_F - Gamma splits the +6.9 divergence into the dropped cohorts'
differential trend B_F and the unidentified composition gap Gamma. Since |Gamma*| = 3.3
is about half of D = 6.9, the credible near-zero reading is overturned only if MORE THAN
HALF of the divergence (3.3 of 6.9 log points) is genuine treatment-effect heterogeneity
rather than differential pre-trend. This makes the "trend vs heterogeneity" ambiguity
quantitative and honors the standing rule (always report both intervals with Gamma*),
completing the Algorithm 1 walk-through in the application. Computed in
carved_application.py (Step 5 block).

## Caveats / notes for the write-up

- Estimated Sigma => first-order (not exact) validity (Remark 3); stated honestly.
- gamma=0 (full conditioning) is the natural default here because the retained
  aggregate is dominated by 2003 (firmly in) and the LATT is stable across selections;
  gamma>0 flips the borderline cohorts and widens the interval without changing the
  near-zero reading.
- Small numeric differences from `fracking_figure.py` come from the pre/post event
  window (PRE_E=[-5..-2], POST_E=[1..4]) and B; the LATT point and the FLCI match the
  paper.
