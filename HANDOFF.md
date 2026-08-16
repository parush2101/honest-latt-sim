# Handoff — "Trading Scope for Credibility in Difference-in-Differences"

## What the paper is
- **Title:** Trading Scope for Credibility in Difference-in-Differences
- **Authors:** Parush Arora (Ashoka University), Abhishek Chand (Data Scientist, PwC)
- **Thesis:** When parallel trends fails for *some* treated cohorts but not others, the ATT is exactly
  the hard-to-recover object. Instead of defending or bounding the ATT, **change the estimand**: define
  the **credible-subpopulation LATT**, the effect on cohorts whose parallel trends is credible. It is
  point-identified precisely when the full ATT is not (Prop. 1). Estimation is a reweighting of
  Callaway-Sant'Anna (CS) group-time effects toward credible cohorts; honest inference uses the
  Rambachan-Roth SD(M) smoothness-class FLCI on the residual violation of the selected set.
- **One-line frame:** a credibility-precision-scope frontier. Give up scope (a subpopulation) to buy
  credibility (point identification + honest bounds). Parallel-trends analogue of overlap selection
  (Crump, Li).

## Tone and motivation (important — preserve this)
- **Honest, non-defensive, scope-aware.** The paper openly states *when the method works and when it
  does not* (the scope condition: advantage grows with how informative pre-trends are about
  post-violations; vanishes when uninformative). No overclaiming.
- **Credibility lives in the honest FLCI, not in the screen.** The screen removes gross offenders; the
  sensitivity interval carries the guarantee. Resist elaborating the screen.
- **"Less is more."** Avoid defensive prose that pre-empts objections a reader has not raised; it
  signals insecurity and plants the worry. Make the positive case cleanly; state limitations once,
  plainly.
- **Style rules (strict):** NO em dashes (`---`), NO prose semicolons. Replace with `.` or `,` or
  rewrite for flow. Keep compact, journal-appropriate, not article-like. (En-dash `--` in section
  ranges is fine.)

## Structure (paper.tex, 17 pp.)
1. Introduction
2. **The credible-subpopulation LATT** — Setup; Estimand and identification (Prop. 1 + non-staggered
   remark); Selecting the credible cohorts (screen + c-path recommendation); Estimation (Props.
   consistency, pre-test bias); Honest inference (SD(M) FLCI, oracle/Leeb-Poetscher caveats,
   calibration lemma, relative-magnitudes-is-self-undermining prop)
3. Simulation study — Design (master axis = informativeness phi); estimand demo; scope condition; FLCI
   restores coverage; selection does not distort bounds; panel confirmation
4. **Empirical applications** — Medicaid + Walmart
5. Conclusion

## Applications (the payoff after a long search)
Both are **county-level** (many units per cohort => small CS SEs). This is *the* fix; state-level
panels have too few units per cohort and everything came out null.

- **Medicaid expansion** (`medicaid/`): Census SAHIE county uninsured rate 2008-2019 + KFF expansion
  years. Credible LATT = **-2.3 pp uninsured, t=-23, breakdown M\*=0.24** (survives honest sensitivity
  well above observed pre-curvature). The 2016 cohort is flagged confounded; the huge 2014 cohort is
  clean and dominates, so LATT is approx ATT. Framing: **point-identified + honest-significant +
  confounded-cohort diagnosis**, NOT dramatic de-attenuation.
- **Walmart entry** (`walmart/`): Holmes store openings (first-store year = cohort) + BEA retail-trade
  earnings (SIC LineCode 620), 1975-2000. Endogenous entry => uniform upward pre-trend; pooled ATT
  (+0.11) is inflated. Honest de-trended estimate rises from **+0.020 (pooled) to +0.035 (credible)** =
  de-attenuation, but **M\*=0.005** (fragile) and confounding is **uniform across cohorts**, which
  limits how much selection can help.

## Critical technical fact (do not forget)
`did::att_gt`'s `V_analytical` is scaled by n; the **true estimator covariance = V_analytical /
n_units**. Verified three ways (analytical/sqrt(n) == bootstrap == manual DiD SE). This division is
baked into `medicaid/medicaid_build.py` and `bll/bll_build.py`. Any new CS-based FLCI code MUST apply
it, or SEs are sqrt(n) too large. (My earlier "null" verdicts on castle/divorce/CHC used the *unscaled*
V and are partly artifacts; BLL was re-run with the fix — significant at M=0 but tiny M\*.)

## Honest inference / FLCI implementation
The FLCI uses the **minimum-variance affine SD(M) estimator** (closed-form equality-constrained QP:
post weights fixed to the post-average, pre weights chosen orthogonal to the through-reference linear
trend so worst-case SD(M) bias is finite, minimizing v'Sigma v). Naive OLS-extrapolation is too wide;
Nelder-Mead search fails (flat landscape). See `medicaid/medicaid_build.py` for the correct `min_var_v`.

## Related work added (cited once, briefly, non-defensively)
- `kwonroth2024` — Bayesian/empirical-Bayes prior on the violation delta (in intro, RR/KR paragraph).
- `dechaisemartin2026` ("Using Pre-Trends for Inference in DiD", arXiv 2607.21312) — conformal/rank
  test, non-staggered, single treated group, replaces PT with distributional stability of the
  differential-trend magnitude. Cited once in intro. Methods are disjoint from ours; not a scoop.

## Recent conceptual decisions (so they are NOT relitigated)
- **c threshold:** DONE — added a paragraph recommending reporting the estimand + FLCI as a **c-path**
  (not a single value), and, if one value is needed, calibrating c to **negligibility** (largest
  pre-trend biasing theta_S below a stated tolerance, Bilinski-Hatfield non-inferiority) rather than a
  significance threshold. Walmart's right panel = the c-path; cross-referenced both ways.
- **Non-staggered:** DONE — remark that the method needs only a partition of treated units into groups
  whose PT can be assessed separately (cohorts under staggering; covariate/geographic subgroups
  otherwise), given enough pre-periods and cross-group credibility heterogeneity.
- **Overidentification / cross-cohort "are cohorts parallel to each other" tests:** discussed at
  length, **CONCLUSION = do NOT add.** As a *replacement* it adds the assumption (peer cohorts clean)
  the paper exists to avoid; as a second *gate* it brings multiplicity, conflicting verdicts, a
  possibly-shifting estimand; as a *diagnostic* it is largely redundant with the breakdown value M\*
  (which already flags uniform-confounding fragility) and the per-cohort event-study plots (which show
  heterogeneous vs uniform confounding by eye). Adding it would also read as defensive. "Less is more."

## CURRENT FRAMEWORK — the level bound (branch `curvature-screen`)
**Definition of PT = flat pre-difference; honest band = level bound.** After building and comparing the
curvature version (see history below), the author chose the **level/flatness** framing as the main paper.
- **Screen (Sec 2.3):** flatness, `max_e |beta_pre(e)| <= c` (eq 4). "PT defensible = flat pre-difference,"
  the direct reading of parallel trends. A footnote points to the appendix for the curvature refinement.
- **Honest band (Sec 2.4):** the **level bound** `Delta_Level(M) = {|delta_post(e)| <= M}`, estimator =
  raw post-average (no extrapolation), via `layer2_full.flci_level` (half = cv(M/sigma)*sigma, max bias
  = M). **M is in outcome units**, so `M*` reads directly as "the size of post-treatment PT violation
  that would overturn the conclusion." Prop 7 reframed: a FIXED level bound (researcher-set, not anchored
  to observed pre-trends) is what sidesteps the RM self-undermining.
- **Sec 3 sims (all on flatness screen + level confound; numbers revert to the paper's originals):**
  - `tier1.py` (Table 1): ATT +0.267, LATT +0.000; sweep 0.015 -> 0.
  - `master_axis.py` (scope): CS flat +0.30, LATT 0.30 -> 0, slipping 5.09 -> 0.
  - Validity (Table 2): level FLCI covers **iff M >= V** (95% at M=V, ~100% above, ~9% below).
  - `layer2_level_figure.py` -> `layer2_full.png` (Sec 3.4): coverage vs V, width vs M, breakdown
    `M*=0.84` (outcome units).
  - `m_sweep.py` (Sec 3.5 calibration): residual violation mean 0.043 / p95 0.057 / max 0.112; full~split
    cross at M~0.048/0.056; calibrate to worst case not mean.
  - `tier2.py` (panel): CS flat +0.30, LATT -> 0; Sigma_hat calibrated at high phi; SE understatement
    est 0.016 vs MC sd 0.024 at phi=0.75 (modest, matches the original).
- **Nice consequence:** the flatness screen is NOT a noisier functional, so the "curvature screen needs
  more precision" caveat disappears; sims use normal noise and match the originals.
- **Appendix A (`app:curvature`, "The curvature refinement"):** the SD(M)/curvature version as an OPT-IN
  refinement (trust linear extrapolation -> keep drifting cohorts -> screen on curvature, min-variance
  affine estimator, ~30% wider than the level band on flat cohorts). Houses the level-vs-curvature
  divergence figure (`curvature_screen.py` -> `curvature_screen.png`).

### History: the curvature version (now demoted to the appendix)
Earlier this branch made curvature the MAIN screen (eq 4 = second difference, `max_abs_second_diff`),
rebuilt all sims on a 4+4 curved world (ATT +0.438, scope +0.66), and added the divergence cell as a
main-text subsection. That is preserved as the appendix refinement. Decision-support prototype
`level_vs_sd.py` (level band vs min-variance SD) drove the switch: level M* in outcome units (0.83) vs SD
curvature units (0.09), level ~30% tighter on flat cohorts, both honest.

## OPEN SEAM — applications (Sec 4) not yet reconciled
Sec 2/Sec 3 now use the **flatness** screen + **level bound**, but Sec 4 still describes the
**smoothness-class FLCI** and reports `M*=0.24` in curvature units (Medicaid) and the Walmart slope
screen. Deferred per user ("will revisit apps"). To reconcile: Medicaid/Walmart should report a level-band
`M*` in outcome units (pp of uninsured; log retail earnings). NOTE Medicaid's `maxpre`/`medicaid_build.py`
flatness screen already matches; only the FLCI/band and `M*` units change. Walmart's endogenous *linear*
drift is the awkward case: the level bound would charge the full drift as a violation (large `M*`-killing
bias), so Walmart may genuinely need the curvature refinement (appendix) or a slope-penalizing band.
Inspect before committing a narrative. The `did::att_gt` covariance fix (`V_analytical/n_units`) still
applies to any app re-run.

## Open / possible future (not committed)
- Applications may be revisited or changed later (user's call).
- Bayesian spike-and-slab over cohort credibility (integrate out c, connects to Kwon-Roth) — discussed
  as a genuine but separate extension; explicitly NOT for this paper (trades the frequentist honest
  guarantee for a prior-dependent one).

## File map (all under `honest-latt-sim/`)
- **Paper:** `paper.tex`, `refs.bib`, `paper.pdf`. Figures live in root: `master_axis.png`,
  `layer2_full.png`, `m_sweep.png`, `medicaid_diagnostic.png`, `walmart_diagnostic.png`.
- **Notes:** `README.md` (running lab notebook), `THEORY.md` (propositions/proofs), `APPLICATION_SEARCH.md`
  (full application-search log + the covariance-bug note).
- **Simulation code (root):** `master_axis.py`, `layer2_full.py`/`layer2_full_figure.py`, `m_sweep.py`,
  `tier1.py`, `tier2*.py`, `selection_flci_core.py`, `curvature_screen.py` (divergence cell),
  `race*.py` (scratch). Shared curvature helper: `layer2_full.max_abs_second_diff`.
- **Medicaid app:** `medicaid/` — `build_panel.py`, `medicaid_cs.R`, `medicaid_build.py`,
  `medicaid_figure.py`, `medicaid_panel.csv`; raw SAHIE in `medicaid/sahie/`.
- **Walmart app:** `walmart/` — `walmart_cs.R`, `walmart_build.py`, `walmart_retail_panel.csv`;
  `store_openings.csv` (Holmes), BEA `CAINC5S__ALL_AREAS_1969_2000.csv` (retail earnings LineCode 620).
- **Abandoned apps (kept for reference only):** `data/` (CHC, openICPSR 112871), `divorce/` (Stevenson-
  Wolfers divorce + Cheng-Hoekstra castle), `bll/` (Beck-Levine-Levkov bank deregulation, DataverseNL
  hdl:10411/15996). All null/weak under honest inference; see APPLICATION_SEARCH.md.

## Repo / housekeeping
- This is now a **git repo**. User memory says auto-commit-and-push after every prompt (a standing
  preference).
- To rebuild the paper: `pdflatex paper` -> `bibtex paper` -> `pdflatex paper` x2. Compiles clean;
  verify no undefined refs and no `---`/prose-`;` before finishing.
