# Handoff — "Trading Scope for Credibility in DiD" (Arora & Chand)

## Where things are
- Repo: `honest-latt-sim/`, branch `curvature-screen`, pushed to github.com/parush2101/honest-latt-sim
- Paper source: `paper.tex` (+ `refs.bib`, `paper.pdf`). Build: `pdflatex → bibtex → pdflatex ×2`.
  Currently **clean at 16pp**.
- macOS gotchas: no `timeout` cmd; `.git/index.lock` sometimes lingers (`rm -f .git/index.lock`);
  cwd can reset after backgrounded commands (use absolute paths / `cd honest-latt-sim`).

## What the paper is
Method: when parallel trends fails for some cohorts, drop them and estimate the
**credible-subpopulation LATT** instead of the ATT. Screen = **flatness** of the pre-trend
(max|beta_pre| <= c). Honest inference = **level bound** Delta^Level(M) (raw post-average, M in
outcome units, breakdown M*). This is the reoriented framework (level bound, not curvature).

Structure: §1 intro, §2 method (setup; selection & estimation; honest inference — Props 1–4 + one
lemma), §3 simulations (estimand demo; scope condition; honest coverage; selection/calibration/panel),
§4 single application, §5 conclusion. No appendix.

## The one application (§4) — shale boom → county house prices
- Treatment: fracking onset by county (year production first rises sharply), from USDA county
  oil+gas production. Outcome: log FHFA county house-price index, 1998–2019. Never-treated = counties
  with negligible production. CS group-time effects, flatness screen, level bound.
- Result: pooled **ATT +6.7% (t=8)** is a housing-bubble + endogenous-onset artifact concentrated in
  the late cohorts (steep pre-trends); the **credible LATT ≈ 0**; divergence **t=5**. Story: the method
  *withdraws* a spuriously reported effect. Written subtly as a plain demonstration — NO mention of the
  dataset search or the frontier tradeoff.
- Figure `fracking_diagnostic.png` (repo root), built by `housing/fracking_figure.py`.
  Panel (b) event study is at illustrative threshold c≈0.03 (stated in text/caption); panel (c) is the
  credibility path.

## Data + scripts (large data gitignored)
- Application: `shale/usda_oilgas_2000_2011.csv` (committed); `housing/fhfa_county.xlsx` (gitignored,
  re-download from FHFA); BEA county income `walmart/CAINC1_*.csv` (gitignored). Figure script:
  `housing/fracking_figure.py`.
- Simulations (in paper): `master_axis.png` (scope), `layer2_full.png` (honest coverage) via
  `master_axis.py`, `layer2_full.py`, `tier1.py`. `m_sweep.py`/`m_sweep.png` exist but the figure was
  CUT from the paper (its results are now prose in §3.5).

## AER:Insights status — QUALIFIES
- Exhibits: **5** (3 figures + 2 tables) — cap is 5. Each ≤ 1 page.
- Words: **~5,490** body (texcount, excl. references) vs the 6,000 cap at 5 exhibits.
- Caveat: unverified whether the journal counts references toward the limit; margin is comfortable.

## Recent editing decisions (already applied)
- Single application only (house-price divergence); dropped Medicaid + Walmart.
- Removed Appendix A (curvature refinement); summarized it in the §2.3 footnote.
- Merged Estimation into "Selection and estimation"; made it estimator-agnostic (not CS-specific).
- Folded old Prop 5 (relative-magnitudes self-undermining) into prose.
- Shorter abstract; merged intro paras 1–2; "large"→"growing" literature; removed repeated
  subpopulation/overlap analogy; removed all prose colons/semicolons (kept math set-builders and
  Keywords/JEL labels); references forced to a new page (`\clearpage`).

## Background (NOT in the paper, for context only)
Behind §4 was an 11-application hunt establishing a robustness↔divergence **frontier**: every robust
case agrees with the ATT (Medicaid, fracking-income), every significant-divergence case has a null/
fragile credible effect (Castle, Divorce, fracking-prices). Documented in `APPLICATION_SEARCH.md` and
per-app `RESULTS.md` (`shale/`, `housing/`, `mj/`, `covid/`, `bll/`). The paper deliberately keeps only
the fracking-prices application and does not narrate this search.

## Open / optional
- `\clearpage` before references leaves whitespace at the end of the conclusion page (expected; revert
  if unwanted).
- Kept `Keywords:` / `JEL classification:` colons and math set-builder colons by choice.
- Fracking numbers are reduced-form event study + county cluster bootstrap; fine as-is, but not run
  through the R `did` package.
