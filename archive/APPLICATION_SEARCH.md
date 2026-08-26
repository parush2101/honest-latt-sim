# Application search log (paused 2026-08-11)

Goal: find a real staggered-DiD application where the credible-subpopulation LATT
demonstrably beats the pooled ATT — ideally a de-attenuated, honest-significant
effect that survives sensitivity bounds. Existence in even ~1 of 10 datasets is a
useful demonstration (with scope disclosed); this is not cherry-picking.

## What a "winning" dataset needs (learned the hard way)
1. A **large, weighty confounded cohort** — big enough to (a) bias the pooled ATT
   and (b) widen the ATT's honest set, so dropping it visibly de-biases AND sharpens.
   Single-state confounded cohorts get averaged down and change nothing.
2. A **long pre-period** (many pre-treatment event-times). Short panels make the
   SD(M) smoothness-class FLCI wide regardless of selection (extrapolation variance).
3. Credible cohorts with a **consistent, real effect** and genuinely flat pre-trends
   (so the affine/honest estimator ≈ the raw estimator).

## Datasets tried
| Dataset | Source | Panel | Cohorts | Result |
|---|---|---|---|---|
| CHC (Bailey–Goodman-Bacon) elderly mortality | openICPSR 112871 | 1959–1998 | ~10 early | Raw LATT −2%, but affine center ~−1%/±wide, **M\*=0**. Credible aggregate pre-trend not flat (−0.33%/yr). Full 1965–80 rollout: SE halves but effect → +0.95%. Covariate (state×year) adj doesn't flatten pre-trend. |
| mpdta (Callaway–Sant'Anna) teen emp | `did` pkg | 2003–2007 | 3 | Too thin; only 3 cohorts, ~1–3 pre-periods; 2007 cohort has a significant pre-period. |
| divorce (Stevenson–Wolfers) female suicide | `bacondecomp` pkg | 1964–1996 | 10 | **Mechanism visible**: corr(maxpre,\|post\|)=0.84; screen drops the +0.58 confounded 1977 cohort. BUT credible LATT ≈ null (pop −0.051 ≈ ATT −0.050; equal −0.001 vs ATT +0.047), **M\*=0**. Confounded cohorts are n=1 (pop-weighting already neutralizes); real effect sits in non-credible cohort 1969. |
| castle (Cheng–Hoekstra) log homicide | `bacondecomp` pkg | 2000–2010 | 5 | Raw ATT +0.09\*, LATT +0.10; screen drops 2009 (pre 0.59, null). BUT 2009 is n=1 → averaged down, so LATT set ≈ ATT set under both SD(M) and RM; short panel → wide bands, **M\*=0**. |

## Update (2026-08-12): BLL bank deregulation + FLCI validation

| Dataset | Source | Panel | Cohorts | Result |
|---|---|---|---|---|
| BLL bank branch deregulation, log Gini | DataverseNL hdl:10411/15996 (free, downloaded to `bll/`) | 1976-2006 (31 yr, 5-6 pre-periods) | 18 state cohorts 1977-1999 | Best STRUCTURAL fit (long panel, many cohorts). Overall CS ATT -0.033. But credible LATT **not significant even raw** (t~-1): state cohorts are small (n=1-6), so CS SEs are large. BLL's famous significance comes from pooled TWFE, which this method avoids. M\*=0. |

**FLCI now validated.** Implemented the minimum-variance affine SD(M) estimator (closed-form
equality-constrained QP in `bll/bll_build.py`: v_post = post-average, v orthogonal to the
through-reference linear trend, minimise v'Sigma v). At M=0 its SE is ~1.1x the raw post-average SE
(0.028 vs 0.025), NOT 4x. So the wide honest intervals and M\*=0 across all datasets are REAL, not an
artifact of a too-wide hand-rolled FLCI. (HonestDiD R package won't compile here -- Fortran deps.)

**Refined, decisive criterion.** The binding constraint is the size of the CS standard error, which is
driven by the number of UNITS PER COHORT. State-level panels give 1-6 units/cohort -> large SEs ->
nothing is significant even before honest inference. The winning application needs **many units per
cohort (county- or firm-level, hundreds each)** so CS SEs are small, plus the earlier requirements
(long panel, a weighty confounded cohort, a real effect).

- FHLT firm-level board reforms (JFE_DID repo): right structure (many firms/cohort) but data is
  author-restricted, NOT free.
- Next free candidates with many units/cohort: Cengiz et al. minimum wage (county-level, openICPSR,
  large download); a county-level disaster/FEMA panel; other AEA-deposited county panels.

## WIN (2026-08-12): ACA Medicaid expansion, county uninsured rate

Free data: SAHIE county uninsured rates 2008-2019 (Census, `medicaid/sahie/`) + KFF expansion
dates. 3146 counties, cohorts 2014 (n=1190 counties), 2015 (191), 2016 (120), 2019 (150),
never-treated (1495). County-level = MANY units/cohort = tiny CS SEs (the fix).

Result (corrected covariance, see bug note): credible LATT (cohort 2014, flat pre-trend maxpre 0.31)
= **-2.30 pp uninsured, t=-23**, and **breakdown M\* = 0.24** -- the SD(M) FLCI EXCLUDES ZERO up to a
substantial smoothness bound. First dataset where honest sensitivity analysis holds. The screen flags
cohort 2016 (maxpre 1.84, anomalous -4.0 effect) as confounded. Event study: flat pre-trends
(e-6..-2 within +/-0.3), sharp drop to -2.6 by e+3. Files: `medicaid/` (build_panel.py, medicaid_cs.R,
medicaid_build.py, medicaid_figure.py, medicaid_diagnostic.png).

## Walmart entry (2026-08-12): de-attenuation, but subtle + fragile

Free data assembled from scratch: Holmes (2011) Walmart store openings (county FIPS + dates,
`walmart/store_openings.csv`) -> county first-entry-year cohort; BEA CAINC5S SIC retail-trade
earnings (LineCode 620) -> county log retail earnings 1975-2000. 2970 counties, 26-year panel,
weighty cohorts (17-126 counties each), 1456 never-treated. Files: `walmart/`.

Result: textbook ENDOGENOUS ENTRY. Treated counties' retail earnings rose ~linearly for 8 yrs
pre-entry (e-8=-0.087 climbing to 0), pooled CS ATT +0.110 (t=18) inflated by this trajectory.
- Confounding is UNIFORM across cohorts (all pre-slopes +0.01..0.03), so credible-subpopulation
  selection can't isolate a "clean" subset -- the uninformative-selection regime for the raw estimate
  (raw LATT ~= raw ATT ~= +0.055).
- BUT de-attenuation appears in HONEST-estimate space: pooled honest (linear-detrended) ATT = +0.020;
  credible (flattest-pre-trend) LATT honest = +0.035 (up to +0.045 at strict thresholds). Robust and
  monotone across screen thresholds. Mechanism: pooling steep-pre cohorts over-corrects the linear
  detrend; restricting to flat-pre cohorts recovers more of the true effect.
- Weakness: breakdown M* = 0.005 (fragile -- effect survives only a small curvature allowance).

## TRADEOFF: Medicaid vs Walmart as THE application
- Medicaid: credible LATT -2.3pp, ROBUST honest significance (M*=0.24), confounded-cohort diagnosis.
  But LATT ~= ATT (2014 clean+dominant) -> no de-attenuation; ATT not really biased.
- Walmart: genuine biased ATT (endogenous entry) + DE-ATTENUATION (honest-LATT > honest-ATT, robust).
  But subtle (only in honest-estimate space; raw LATT~=raw ATT) and fragile (M*=0.005).

## BUG FOUND + FIXED: did V_analytical scaling (affects ALL CS datasets)

`did::att_gt`'s `V_analytical` is the influence-function outer product, scaled by n; the estimator
covariance is `V_analytical / n_units`. Verified on att(2014,2014): analytical se 3.97 / sqrt(3146)
= 0.072 = bootstrap se = manual DiD se. My earlier bll/castle/divorce/CHC FLCIs used the UNSCALED V,
so intervals were sqrt(n) too WIDE and t-stats sqrt(n) too small.
- **BLL corrected**: raw/honest effect is now significant at M=0 (t=-3 to -6.8), NOT the "t~-1"
  reported earlier -- that was a scaling artifact. But BLL breakdown M\* is tiny (~0.0015): significant
  under exact linearity, fragile to smoothness violations. Borderline, not a clean win.
- **castle / divorce / CHC**: earlier "M\*=0 / null" conclusions were partly the same artifact and
  should be RE-RUN with the /n fix before trusting. (divorce/castle LATT were ~null in POINT estimate
  regardless, so likely still weak; CHC/castle may change.)

## Root-cause finding
Across all four: reproducible canonical datasets have **small confounded cohorts**
(1–2 states), so the method has little to bite on — dropping them barely moves the
point estimate OR the aggregate's max pre-trend. Combined with short/noisy panels,
honest smoothness-class inference includes zero regardless of selection. The raw
point estimates show the mechanism (LATT flags/drops confounded cohorts); the honest
CONFIDENCE statement does not exclude zero.

## Candidate directions for "later" (untried, likely need download)
- Minimum-wage state panels (Cengiz et al.; long panel, many state cohorts, known
  bad-actor cohorts) — needs replication package.
- Wolfers 2006 divorce RATE (`drate`) — different outcome in the same data, but
  non-monotone dynamics fight the SD(M) restriction (noted, not recommended).
- Any policy panel with ONE large region/state cohort adopting during a known
  contemporaneous shock (the ideal "weighty confound").

## Open TODO before any application ships
- Reconcile propagated event-study SE (~0.14 in Python) vs R's group-aggregated SE
  (0.036). Aggregations differ; my specific interval widths are not final until fixed.
  Qualitative conclusions (LATT≈ATT in point + aggregate pre-trend) are unaffected.
- paper.tex Section 5 STILL contains the original CHC application with the −2% /
  "flat pre-trend" / "less-attenuated" claims, which this analysis shows are fragile
  (M\*=0; credible aggregate pre-trend −0.33%/yr). Not yet rewritten — decide later.

## Reproduce
- CHC extract rebuilt into `data/chc_slim.csv` (+ `chc_slim2.csv` w/ stfips) from
  `~/Downloads/112871-V1/...aer_data.dta`. Screen generator was never in the repo;
  documented reconstruction (pop-weighted, pre −4..−1) reproduces the original
  flagged {1965,1970,1973}.
- divorce: `Rscript divorce/divorce_cs.R` then `python3 divorce/divorce_build.py`
- castle:  `Rscript divorce/castle_cs.R`  then `python3 divorce/castle_build.py`
