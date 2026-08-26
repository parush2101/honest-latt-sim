# Theory & Paper Skeleton

Working title: *A Credible-Subpopulation Approach to Parallel Trends*
(a.k.a. "the honest LATT paper")

---

## 1. One-sentence contribution

HonestDiD keeps the ATT as the target and reports a *set*; we **change the estimand** to
a credibly-identified subpopulation effect (a local ATT) that is **point-identified when
the ATT is not**, and give honest inference for it. It is not a better estimator — it is a
different, defensible estimand choice.

## 2. Positioning

- **Lineage:** same move as LATE (Imbens–Angrist), optimal-subpopulation ATE (Crump, Hotz,
  Imbens & Mitnik 2009), and overlap weights / ATO (Li, Morgan & Zaslavsky 2018): give up
  the population target for a credibly-identified subpopulation one. We are the
  staggered-DiD / parallel-trends instance of that principle.
- **Spectrum (the paper's framing figure):** three points on a credibility–precision–scope
  frontier.
  - CS / heterogeneity-robust: point estimate of the ATT, fragile (assumes PT).
  - HonestDiD (RR): set for the ATT, robust (assumes bounded violation), agnostic/wide.
  - **Ours:** point (+ honest set) for a *subpopulation* effect, under an
    informative-pre-trends / smoothness assumption. Buys back a sharp answer to a smaller
    question.

## 3. Setup & notation

Standard staggered DiD (Callaway–Sant'Anna). Cohorts g, group-time effects ATT(g,t),
event-study coefficients β = τ + δ (Rambachan–Roth decomposition: τ causal, τ_pre = 0 by
no-anticipation; δ differential trend; parallel trends ⇔ δ_post = 0).

Selection / weighting rule R produces a subpopulation S (or weights w_g). Two regimes:

- **Regime A — ex-ante / split selection:** S is a function of pre-determined covariates,
  economic reasoning, or an independent data split. **S is independent of the estimation
  sample.**
- **Regime B — data-driven selection:** S is chosen from the *realized* pre-trends
  (β̂_pre). **S depends on the estimation sample.**

Target estimand: the **credible-subpopulation LATT**
  θ_S = Σ_{g∈S} w_g · ATT(g, ·) / Σ_{g∈S} w_g.

## 4. Theoretical claims (with honest proof status)

**Proposition 1 (Estimand & identification).** If parallel trends holds for the cohorts in
S (i.e. δ_g,post = 0 for g ∈ S), then θ_S is point-identified by the CS aggregation
restricted to S, *even if the full ATT is not identified* (because excluded cohorts violate
PT). The excluded cohorts' violations do not enter θ_S. — *Status: straightforward
(definitional + CS identification). The content is precision about what "credible for S"
means.*

**Proposition 2 (Consistency & efficiency).** Under Regime A, θ̂_S is consistent for θ_S;
under full parallel trends it is consistent for the ATT, with an efficiency loss relative to
CS/BJS. — *Status: standard, from CS asymptotics. (Say "consistent," not "unbiased.")*

**Proposition 3 (Pre-test bias, Regime B).** Under data-driven selection, θ̂_S carries a
finite-sample selection (pre-test) bias that is O(·) vanishing as pre-test power → 1 (large
per-cohort samples). — *Status: follows from the post-selection / pre-test-bias literature
(Roth 2022); state the rate. This is the honest caveat, confirmed by the sample-size sweep
(Tier 1) where LATT bias → 0.*

**Proposition 4 (Honest inference on the selected set — LOAD-BEARING).** Given S and a
restriction δ_S,post ∈ Δ on the residual differential trend of the selected aggregate, the
SD(M) FLCI (resp. ARP hybrid) delivers uniformly valid (1−α) confidence sets for θ_S.
  - **Regime A:** immediate — S ⟂ estimation sample, so Rambachan–Roth applies verbatim to
    the selected-aggregate event study. *Provable.*
  - **Regime B:** the simulations find **no systematic selection distortion** of the FLCI
    (full-data ≈ split coverage; center distortion small and, if anything, makes full-data
    *less* biased than split). *Simulation-supported; a proof needs post-selection-inference
    machinery and is open. The paper's safe theoretical claim is Regime A; Regime B is the
    empirically-validated extension.*

**Lemma 5 (Calibration to the random selected set — NOVEL).** In Regime B the selected set
is random, so the residual curvature of its aggregate is a random variable. For unconditional
coverage, M must bound that residual curvature at (roughly) an upper quantile, not its mean:
a **mean-calibrated M undercovers**. Sampling slack in the FLCI means M need not reach the
strict max, but must clearly exceed the mean. — *Status: new; clean to state and prove given
the selection rule's induced distribution of residual curvature. Confirmed by the M-sweep
(mean 0.031 → ~88%; 95% at M ≈ 0.039 < max 0.050).*

**Proposition 6 (Relative-magnitudes is self-undermining — NOVEL negative result).** A
Δ^RM(M̄) restriction anchors the identified-set width to the *observed* pre-trend magnitude.
When pre-trends are uninformative about post-treatment violations (weak link), the observed
pre-trends are small, so Δ^RM produces intervals whose width → 0 while the true violation
does not — hence undercoverage / false precision. A structural bound (SD(M), or an external
magnitude) does not share this defect because it keys off smoothness, not pre-trend
magnitude. — *Status: new; provable via the RM identified-set width formula. Confirmed by the
Layer 2 prototype.*

**Proposition 7 (Decision-theoretic dominance region — the FRAMING).** Under a stated loss
(e.g. MSE, or a decision loss over a hypothesis), there is a region of the
(pre-trend-informativeness × sampling-precision) space where the point-identified LATT has
lower risk than the set-valued ATT, and a complementary region where it does not. — *Status:
the intellectual justification for the whole approach; needs an Armstrong–Kolesár-style
decision setup. This is what turns "we redefine the estimand" into "here is when you should."
The master-axis figure is its empirical shadow.*

### What is genuinely novel vs. applied

- Novel: the **estimand-choice framing** for staggered DiD (P1, P7), the **integrated
  selection+HonestDiD validity** (P4 Regime B), the **calibration lemma** (L5), the
  **RM self-undermining critique** (P6).
- Applied / standard: consistency (P2), the FLCI itself (P4 machinery = Rambachan–Roth),
  pre-test bias (P3 = Roth 2022).

## 5. Paper skeleton

1. **Introduction** — the pre-trends dilemma; the two responses (agnostic set vs credible
   subpopulation); one-sentence contribution; the spectrum figure.
2. **Setup** — staggered DiD, CS aggregation, RR decomposition, the two selection regimes.
3. **The credible-subpopulation LATT** — estimand definition, identification (P1),
   interpretation (who is the subpopulation? — the LATE-style burden), positioning vs
   LATE / overlap weights.
4. **Estimation** — the reweighted estimator, consistency & efficiency (P2), Regime-B
   pre-test bias and its vanishing (P3).
5. **Honest inference on the selected set** — Δ restrictions, the FLCI/ARP applied to the
   selected aggregate, validity (P4), the calibration lemma (L5). Recommend SD/structural
   over RM, with the self-undermining result (P6).
6. **When to prefer the LATT (the scope condition)** — the decision-theoretic dominance
   region (P7); the master-axis result; honest limits (uninformative pre-trends ⇒ no free
   lunch; the assumption lives in Δ / the economics).
7. **Simulations** — Tiers 1–2, master-axis, Layer 2 validation, selection×FLCI,
   M-calibration.
8. **Empirical application(s)** — re-analyze a staggered-DiD paper; report the LATT with its
   honest set and the breakdown value; contrast with CS and HonestDiD-on-the-ATT.
9. **Conclusion.**

## 6. Status: done vs. remaining

- **Done (simulation):** spine (Tiers 1–2), scope condition, Layer 2 FLCI validated,
  selection×FLCI non-distortion, M-calibration, RM critique. Layer 1 selective inference
  provisionally **cut** (splitting suffices; distortion small).
- **Remaining (theory):** write P1–P7 as formal statements; prove L5 and P6 (clean);
  P4 Regime A (immediate) and state Regime B as validated-conjecture; build the P7 decision
  setup (the real intellectual work).
- **Remaining (empirical/software):** full panel pipeline + an application (writing-stage).
