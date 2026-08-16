# Shale/fracking application (built 2026-08-16, Option A hunt)

Treatment: fracking onset by county, dated from USDA ERS county oil+gas production
(2000-2011, H_Growth >$20M increase counties), onset = first year combined BOE
(oil_bbl + gas_mcf/6) reaches baseline + 25% of (peak-baseline). 218 boom counties,
onset 2002-2011, 12-35 counties/cohort (TX 67, PA 20, OK 18, KS 17, WV 13...).
Outcome: BEA CAINC1 county income, 1998-2019. Never-treated = 2677 non-boom,
non-declining counties. Comparison = reduced-form cohort event study vs NT mean;
cluster bootstrap over counties (fixed credible set).

## Per-capita personal income (LineCode 3)
- ATT (9 cohorts 2003-2011) = +0.057 (t=6.3), M*=0.039
- Credible LATT {2003,2006,2008} (screen c=0.03) = +0.044 (t=3.3), M*=0.018
- M*=0.018 < retained pre-trend band (~0.025) -> FRAGILE

## Total personal income (LineCode 1)  [stronger outcome: royalties + in-migration]
- ATT = +0.071 (t=7.1), M*=0.052
- Credible LATT {2003,2005,2006,2008} (c=0.03) = +0.060 (t=4.8), M*=0.035
- Retained pre-trend band <=0.028 (2003=.019,2005=.021,2006=.028,2008=.016)
- M*=0.035 > band -> ROBUST
- BUT divergence ATT-LATT = 0.011, se=0.011, t=0.99 -> NOT statistically significant
- Screen path stable: LATT ~0.058-0.060 for c in [0.025,0.04]; 0.074 at c=0.05 (adds 2010)

## Verdict
8th application. Confirms the robustness<->divergence frontier again, in a fresh,
large-sample, marquee setting:
- ROBUST (first robust case besides Medicaid; bigger effect, more famous than Medicaid)
- but divergence is NOT significant (like Medicaid: LATT is a modestly-lower,
  statistically-indistinguishable version of ATT).
Value: a second, stronger ROBUST application for the paper (better lead than Medicaid),
and independent corroboration of the frontier tradeoff. Does NOT deliver robust+significant
divergence -- consistent with the structural argument that confounding strong enough to
bias the ATT also contaminates retained cohorts.
