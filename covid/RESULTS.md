# COVID-confounded county shot (10th app, 2026-08-16) -- NEGATIVE

Design: recreational cannabis legalization (staggered, weighty 2021 wave: MT/NJ/NY/VA/NM/CT
= 314 counties) -> county employment (BLS LAUS annual, log Employed, 2015-2023).
Never-treated = 2019 non-legalizing counties. Hypothesis: 2021 cohort has COVID (2020) in
its pre-period -> rebound bias -> non-flat pre-trend -> screen drops it -> divergence.

Result: DOES NOT WORK.
- Pooled ATT = -0.008 (cannabis has ~no county-employment effect). LATT -0.005..-0.032 (~0).
- 2021 cohort (ref year 2020 = COVID trough) has FLAT pre-trend (maxpre 0.011) and is KEPT.

Structural reason (the sharp lesson):
COVID is a COMMON shock -- it hit never-treated counties too -- so the DiD against the
never-treated comparison group DIFFERENCES IT OUT. A macro shock only biases a cohort's
DiD if the TREATED group is DIFFERENTIALLY exposed relative to controls. COVID isn't
(cannabis states aren't systematically more COVID-exposed), and differential exposure
isn't cleanly cohort-aligned or pre-trend-detectable anyway. So the feature that made
COVID attractive (huge, 2020-aligned, weighty) is exactly what makes it invisible to the
method: it cancels in the comparison. This is the Kahn-Lang contemporaneous-confound /
common-time-shock point; not a fit for the credible-subpopulation method.

## Bottom line across 10 applications
The robustness<->divergence frontier holds with no exception. For SIGNIFICANT divergence
the confound must be (a) treated-group-specific (survive differencing), (b) cohort-aligned
& pre-trend-detectable (so the screen catches it), (c) weighty (bias the ATT), and (d) leave
clean cohorts with a real effect. No real dataset satisfies all four: the confounds that are
big enough to bias the ATT significantly are either common (differenced out, COVID), geographic
not cohort-timed (fentanyl), or contaminate the retained cohorts too (Castle/Divorce fragile).
The one robust county case with a real effect (fracking, Medicaid) has clean & confounded
cohorts too similar -> divergence not significant. THIS IS THE PAPER'S STRUCTURAL THESIS.
