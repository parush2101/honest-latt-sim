# Fracking -> county house prices (FHFA HPI) -- 11th app, 2026-08-16
# THE SIGNIFICANT DIVERGENCE (first in 11 apps)

Treatment: fracking onset (same USDA-based cohorts as shale/). Outcome: log FHFA
county all-transactions HPI (2000 base), 1998-2019. 134 boom + 2104 NT counties.

Cohort event study (reduced-form vs NT, cluster bootstrap over counties):
  g     n  maxpre  post-eff
  2003 15  0.008  -0.065
  2004 13  0.022  +0.016
  2005 10  0.006  +0.070
  2006 14  0.040  +0.090
  2007 11  0.070  +0.092
  2008 13  0.077  +0.129
  2009 10  0.040  +0.066
  2010 22  0.079  +0.120
  2011 10  0.110  +0.076

- ATT (all 9) = +0.067  (t=7.8)  "fracking raised house prices 7%"
- Credible LATT (flat-pre {2003,04,05}, c=0.03) = -0.002 (t=-0.1) ~ ZERO
- DIVERGENCE ATT-LATT = +0.069, se=0.014, t=4.94, 100% of bootstraps > 0  <-- SIGNIFICANT
- Credibility path: LATT rises 0 -> 0.030 -> 0.066 as c relaxes 0.03->0.08 (admits bubble cohorts)

Mechanism: late cohorts (2006-2011) had prices ALREADY rising pre-onset (steep pre-trends,
the 2003-06 housing bubble + endogenous onset). Their measured "effect" is pre-existing
appreciation, not fracking. Flat-pre cohorts (2003-05) show ~0. Credible answer: fracking
had ~no net effect on house prices -- consistent with the fracking-HPI literature
(amenity demand vs groundwater/disamenity roughly cancel; Muehlenbachs et al.).

## Which horn of the frontier
This is SIGNIFICANT DIVERGENCE but the credible effect is NULL (condition 4 fails: clean
cohorts don't retain a real price effect). So it's "method overturns a spurious ATT",
NOT "robust nonzero divergence". Still the strongest divergence in 11 apps (t=4.9 vs
Castle/Divorce fragile, fracking-income t=0.99).

## THE TWO-OUTCOME FRACKING STORY (both margins, one setting)
- Fracking -> INCOME:  robust (+6%, M*=0.035), divergence NOT significant (t=0.99)
  -> method CONFIRMS a real effect.
- Fracking -> PRICES:  divergence decisive (t=4.9), credible effect null
  -> method OVERTURNS a spurious ATT (housing-bubble artifact).
Same treatment/cohorts/screen, opposite honest conclusions. Ideal Section 4.

## Data-access note (the predicted wall)
Best-fit divergence candidates (RTC/gun laws -> county crime; Clean Air Act nonattainment)
are DATA-GATED: openICPSR county UCR needs login; Donohue's comprehensive RTC study is
state-level (units wall). This is the "publication/data pre-selection" prediction biting:
messy settings where the method bites have gated data; clean-download settings lack the
confound structure. Fracking-HPI worked because the confound (housing bubble) is external,
cohort-aligned via onset timing, and detectable -- and I already had the onset dates.
