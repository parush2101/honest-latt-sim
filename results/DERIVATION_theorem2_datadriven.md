# Theorem 2 under data-driven selection: the derivation attempt and its outcome

**Question (handoff #3 / Refine #1, #16).** Theorem 2 proves a width-dominance
threshold for the *ex-ante* (fixed-length FLCI) case. The submitted text added, for
the *data-driven* (carved) case, that "the threshold holds in expectation under the
same variance ordering." That expected-length comparison was never derived. Derive
it, or restrict Theorem 2 to ex-ante selection.

**Outcome.** The derivation does not yield a dominance theorem; it yields an
**impossibility**. The expected-length comparison is ill-posed because the carved
interval has infinite expected length, and no quantile-length dominance survives
either, because the length ordering flips against carving in the exact regime that
motivates it. The clean, defensible resolution is therefore to restrict Theorem 2's
width statement to ex-ante selection — now backed by a proved no-go, not a choice —
and to compare the data-driven intervals by **coverage** (both valid) plus a robust
**median** length summary in §5.6.

---

## Proposition (No expected-length dominance under data-driven selection)

*Maintain Assumption 2. Under data-driven selection the carved interval
$\mathcal{C}_{1-\alpha}(\hat S)$ of Theorem 1 is the polyhedral inversion of the
truncated-normal pivot $F^{[\mathcal V^-,\mathcal V^+]}_{\theta_S,s_S^2}$. Then for
every $\beta$ and every $S\neq\varnothing$,*
$$
\mathbb{E}\!\left[\,\bigl|\mathcal{C}_{1-\alpha}(S)\bigr| \;\middle|\; \hat S = S\right] \;=\; \infty .
$$
*Consequently the width comparison of Theorem 2 is well-posed only for the ex-ante
fixed-length intervals, and no expected-length ordering between the carved LATT
interval and the ATT interval exists under data-driven selection.*

**Proof.** By Theorem 1, conditional on $\{\hat S=S\}$ and the orthogonal remainder
$r$, the estimate is truncated Gaussian,
$\hat\theta_S \mid \{\hat S=S\},r \sim \mathcal{TN}(\theta_S, s_S^2,[\mathcal V^-,\mathcal V^+])$,
and $\mathcal{C}_{1-\alpha}(S)$ is obtained by inverting the truncated-normal pivot
$F^{[\mathcal V^-,\mathcal V^+]}_{\theta_S,s_S^2}(\hat\theta_S)$ in $\theta_S$. This
is exactly the confidence set of Lee et al. (2016) for the mean of a normal observed
under a polyhedral (hence interval, after conditioning on $r$) truncation.
Kivaranović and Leeb (2021, Thm 1) show that this set has infinite expected length
whenever the truncation is one-sided with positive probability, i.e. whenever
$\Pr(\mathcal V^-=-\infty)+\Pr(\mathcal V^+=+\infty)>0$ conditional on selection. Under
the flatness screen a retained cohort's binding constraint is two-sided only when
*both* $|\hat\beta_{g,\mathrm{pre}}(e)+\omega_g|\le c$ faces are active; generically
one face binds and the induced bound on $\hat\theta_S$ is one-sided, so the event has
positive probability and the hypothesis holds. Hence the conditional expected length
is infinite. Averaging over $S$ preserves it. $\qquad\blacksquare$

**Randomization does not remove it.** For every finite $\gamma\ge 0$ the procedure
hard-conditions on $\{\hat S=S\}$ together with the active screen coordinates, so the
pivot remains truncated normal and the proposition applies. Only $\gamma\to\infty$
escapes, by sending every retention probability to zero (empty selection with
probability approaching one), leaving nothing to invert.

## No quantile-length dominance either

A median-length comparison is well-posed but does not order the two intervals in
general. Heuristically, the carved length is near the untruncated $2z\,s_S$ when
$\hat\theta_S$ lies interior to $[\mathcal V^-,\mathcal V^+]$ and diverges as it
approaches a boundary. Interior draws dominate — hence a finite median — only when
$\hat\theta_S$ is weakly coupled to the binding pre-period constraints. But carving is
*needed* (the naive interval undercovers) precisely when pre/post correlation is high
and cohorts sit near the threshold, which is when $\hat\theta_S$ is *strongly* coupled
to those constraints and the truncation bites even at the median. The median length
ordering therefore flips against carving in the high-distortion regime that motivates
it, and coincides with the naive interval (where carving is unnecessary) in the
low-distortion regime. No fixed threshold on the dropped violation reproduces the
clean $V^\ast$ of the ex-ante case.

## Numerical corroboration (`carved_gamma.py`)

Selection-isolation design (all cohorts clean in post, target $=1$; $G=8$, four
pre-periods, AR(1) $\rho=0.85$, borderline cohorts at $0.36$ below $c=0.40$):

| $\gamma$ | carved cov | naive cov | split cov | **median** carved | median split | p90 carved | empty % |
|---|---|---|---|---|---|---|---|
| 0.0 | 95.4 | 90.6 | 94.6 | 0.545 | 0.444 | 2.47 | 0.0 |
| 0.5 | 95.8 | 92.2 | 95.1 | 0.555 | 0.444 | 2.24 | 0.0 |
| 2.0 | 95.1 | 94.1 | 95.2 | 0.621 | 0.512 | 2.52 | 0.7 |
| 8.0 | 95.3 | 94.8 | 94.5 | 0.855 | 0.627 | 3.17 | 10.6 |

Infinite-mean signature (fixed $\gamma=0.5$, growing replications): median stable at
$\approx 0.556$ while the sample max is erratic and unbounded (4047, 1817, 1426, 3162)
and the sample mean fails to converge (6.2, 2.6, 2.9, 2.2). Carving is **valid** at
every $\gamma$ (coverage $\approx 95\%$) where the naive interval undercovers; its
median length is **comparable to, not shorter than**, a proper $\sqrt2$ sample-split;
and its edge over splitting is deterministic full-sample use, not width.

## Recommended paper edits

1. **Theorem 2 / §4.** State width dominance for ex-ante selection only. Replace the
   sentence "the threshold holds in expectation under the same variance ordering" for
   the carved case with a pointer to the Proposition above: under data-driven
   selection the carved interval has infinite expected length, so no width-dominance
   threshold is well-posed; the honest comparison is coverage plus median length.
2. **Add the Proposition** (short; proof is two lines plus the KL citation) either in
   §4 or as an appendix remark, and add **Kivaranović & Leeb (2021)** to `refs.bib`.
3. **§5.6.** Report the `carved_gamma.py` sweep: carved coverage $\approx 95\%$ vs
   naive undercoverage across $\gamma$, with median (not mean) length, and state the
   infinite-expected-length fact explicitly. This is the exhibit Refine #5 asks for.
4. **Prop 4 / Corollary 1 language.** Where "length" or "expected length" is used for
   the carved interval, switch to "median length" or "coverage," and keep
   "expected/fixed length" only for the ex-ante case.

## Reference to add

Kivaranović, Danijel and Hannes Leeb, "On the Length of Post-Model-Selection
Confidence Intervals Conditional on Polyhedral Constraints," *Journal of the American
Statistical Association*, 2021, 116 (534), 845–857.
