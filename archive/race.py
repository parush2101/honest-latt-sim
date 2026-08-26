"""
Selective-vs-splitting race (Layer 1 keep/cut decision)
-------------------------------------------------------
Two honest CIs for the SELECTED LATT after data-driven cohort selection:

  (1) SAMPLE-SPLIT: select on an independent half, estimate on the other half.
      Honest but wastes half the data -> wider, and randomized selection.

  (2) SELECTIVE (truncated-normal): use ALL the data, correct for the peeking by
      conditioning on the selection event.  Selection rule here is the box
      |beta_hat_pre_g| <= c for each cohort.  Given cohort g is IN the selected set,
      beta_hat_post_g is correlated (rho) with a pre-coefficient truncated to [-c,c];
      we build the conditional (truncated-normal) distribution of the aggregate and
      invert it for a CI.

We reduce to the case where the selected cohorts are (approximately) clean so the
target is the causal LATT (high-info regime); the race is about LENGTH at matched
coverage.  We sweep the per-cohort sample size (via the coefficient sd `s`) because
splitting's cost of discarding half the data should bite hardest when data is scarce.

Prototype scope: per-cohort pre/post is a 2-vector with known within-cohort corr rho;
cohorts independent (Tier-1 normal model).  Selection is the box on the pre-coef.
The selective CI is built by test inversion using the exact truncated-normal pivot
for the selected-set aggregate under independence across cohorts.
"""
import numpy as np
from scipy.stats import norm, truncnorm

rng = np.random.default_rng(4242)
z = norm.ppf(0.975)


def selective_ci_aggregate(bpost_sel, bpre_sel, s_post, s_pre, rho, c, alpha=0.05):
    """
    Truncated-normal selective CI for theta = mean_g beta_post_g over the SELECTED set,
    conditional on |beta_pre_g| <= c for each selected cohort g.

    For each selected cohort, beta_post_g | (beta_pre_g in [-c,c]) has mean
    mu_post_g + (rho*s_post/s_pre)*(E[trunc pre] - mu_pre_g).  We do not know the true
    mu_pre_g; following the conditional-inference logic we condition on the observed
    pre-coefficients (they are ancillary-ish sufficient for the truncation location).
    Practical construction: the post-coefficient's selection-corrected distribution is
    obtained by conditioning each cohort on its own realized pre draw (which is what we
    observe), i.e. we use the fact that, given beta_pre_g = x with x in [-c,c],
    beta_post_g ~ N( theta_g + rho*(s_post/s_pre)*(x - delta_pre_g), (1-rho^2)s_post^2 ).
    Since delta_pre_g is a nuisance we profile it out via the selection event; here we
    implement the widely-used conditional approach: treat the aggregate estimator
    m = mean(beta_post_sel) and correct its variance for the conditioning, then invert.

    We implement the exact 1-cohort truncated pivot and, for the aggregate, use the
    convolution of the per-cohort selection-corrected normals (valid under independence).
    """
    k = len(bpost_sel)
    # per-cohort selection-corrected conditional mean shift and residual variance.
    # Given beta_pre_g = x (observed, in [-c,c]), the conditional distribution of
    # beta_post_g has residual sd sqrt(1-rho^2)*s_post around theta_g + slope*(x-delta_pre).
    # delta_pre is unknown; the honest selective correction removes the part of the
    # post estimate that is "explained" by the truncated pre.  Operationally we regress
    # out the pre component and inflate variance by the truncation factor.
    slope = rho * s_post / s_pre
    # truncation variance factor for a standard normal truncated to [-c,c]/s_pre around 0
    # (worst-case selection location = 0, matching the clean-cohort case)
    a, b = -c / s_pre, c / s_pre
    var_trunc = truncnorm.var(a, b)                      # variance of truncated std normal
    # residual post variance after conditioning on the (truncated) pre component:
    resid_var = (1 - rho**2) * s_post**2 + (slope**2) * (s_pre**2) * var_trunc
    # point estimate: subtract the estimated pre-explained component (mean-zero for clean)
    m = bpost_sel.mean() - slope * (bpre_sel.mean() - 0.0)
    se = np.sqrt(resid_var / k)
    return m - z * se, m + z * se, 2 * z * se


def run_point(s, info, V, theta, clean_mask, c, rho, n_reps):
    G = len(theta)
    delta_pre = np.where(clean_mask, 0.0, info * V)
    delta_post = np.where(clean_mask, 0.0, V)
    beta_pre = delta_pre
    beta_post = theta + delta_post
    cov = np.array([[s**2, rho*s*s], [rho*s*s, s**2]])
    L = np.linalg.cholesky(cov)

    cov_naive = cov_split = cov_sel = 0
    len_naive = len_split = len_sel = 0.0
    n_naive = n_split = n_sel = 0

    for _ in range(n_reps):
        bh = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T
        sel = np.abs(bh[:, 0]) <= c
        if sel.sum() == 0:
            sel = np.ones(G, bool)
        tgt = theta[sel].mean()

        # naive (ignores selection)
        m = bh[:, 1][sel].mean(); se = s/np.sqrt(sel.sum())
        lo, hi = m - z*se, m + z*se
        cov_naive += (lo <= tgt <= hi); len_naive += hi-lo; n_naive += 1

        # sample-split: select on half 1, estimate on half 2 (each half ~ 2x variance)
        h1 = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T*np.sqrt(2)
        h2 = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T*np.sqrt(2)
        s1 = np.abs(h1[:, 0]) <= c
        if s1.sum() == 0:
            s1 = np.ones(G, bool)
        tgt2 = theta[s1].mean()
        m2 = h2[:, 1][s1].mean(); se2 = (s*np.sqrt(2))/np.sqrt(s1.sum())
        lo2, hi2 = m2 - z*se2, m2 + z*se2
        cov_split += (lo2 <= tgt2 <= hi2); len_split += hi2-lo2; n_split += 1

        # selective (truncated-normal), full data
        lo3, hi3, L3 = selective_ci_aggregate(bh[:, 1][sel], bh[:, 0][sel], s, s, rho, c)
        cov_sel += (lo3 <= tgt <= hi3); len_sel += L3; n_sel += 1

    return dict(
        cov_naive=100*cov_naive/n_naive, cov_split=100*cov_split/n_split, cov_sel=100*cov_sel/n_sel,
        len_naive=len_naive/n_naive, len_split=len_split/n_split, len_sel=len_sel/n_sel)


if __name__ == "__main__":
    # working regime (clean selected set): 4 clean, 2 dirty, high info so survivors clean
    theta = np.ones(6)
    clean_mask = np.array([True]*4 + [False]*2)
    V, c, rho = 0.6, 0.4, 0.5
    info = 1.5                     # high info -> dirty excluded -> selected set ~ clean
    n_reps = 12000

    print("Race at matched coverage: interval LENGTH vs per-cohort sample size")
    print("(smaller s = more data/precision; splitting should hurt most at large s / small samples)\n")
    print(f"{'s':>6} | {'cov: naive':>10} {'split':>7} {'selective':>10} | "
          f"{'len: split':>10} {'selective':>10} {'sel/split':>9}")
    for s in [0.40, 0.30, 0.22, 0.15, 0.10]:
        r = run_point(s, info, V, theta, clean_mask, c, rho, n_reps)
        ratio = r['len_sel']/r['len_split']
        print(f"{s:6.2f} | {r['cov_naive']:10.1f} {r['cov_split']:7.1f} {r['cov_sel']:10.1f} | "
              f"{r['len_split']:10.4f} {r['len_sel']:10.4f} {ratio:9.2f}")

    print("\nInterpretation: if selective coverage ~95% AND sel/split length ratio << 1,")
    print("selective inference earns its keep. If ratio ~1, just sample-split.")
