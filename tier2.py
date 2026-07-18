"""
Tier 2 simulation: full staggered PANEL micro-data + real CS-style estimation.

Difference from Tier 1: we generate individual panel data, estimate the cohort-level
pre/post coefficients with actual difference-in-differences contrasts, and estimate
their covariance Sigma_hat FROM THE DATA (Tier 1 used a known Sigma). A shared
never-treated control group induces real cross-cohort correlation that the
aggregation standard error must account for.

DGP (untreated potential outcome):
    Y_it(0) = alpha_i + phi_t + m_g(t) + eps_it
  clean cohort:  m_g(t) = 0                              (parallel trends holds)
  dirty cohort:  m_g(t) = (info*V)*(t-(g-1)) + (1-info)*V*1{t>=g}
                 -> pre-trend  ~ info*V   (visible in pre-periods)
                 -> post-trend ~ V        (part invisible in pre when info<1)
Treatment effect: Y_it(1) = Y_it(0) + theta_g * 1{t>=g},  theta_g = 1 (bias isolation).

CS-style contrasts vs never-treated control, base period b = g-1:
    beta_pre_g  = (Ybar_g,g-2 - Ybar_g,g-1) - (Ybar_c,g-2 - Ybar_c,g-1)   -> estimates delta_pre
    beta_post_g = (Ybar_g,g   - Ybar_g,g-1) - (Ybar_c,g   - Ybar_c,g-1)   -> estimates theta_g + delta_post
Sigma_hat: analytic influence-function covariance (treated part block-diagonal;
shared control part dense -> cross-cohort correlation).

Selection: |beta_hat_pre_g| <= c.   CS = mean beta_post over ALL; LATT = mean over SELECTED.
Aggregation SE uses Sigma_hat (incl. cross-cohort covariance among selected cohorts).
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20240718)

# ---- design ----
cohorts = np.array([4, 5, 6, 7, 8, 9])          # staggered treatment start periods
clean_mask = np.array([True, True, True, False, False, False])
G = len(cohorts)
T = 12
N_g = 250                                       # units per treated cohort
N_c = 700                                       # never-treated control units
V = 0.6
theta = np.ones(G)                              # true effect = 1 for all -> true ATT = true LATT = 1
sigma = 1.5                                      # idiosyncratic noise sd
phi = np.linspace(0, 1.0, T)                    # common time effects (cancel in DiD)
c = 0.40
z = norm.ppf(0.975)
TRUE = 1.0


def m_path(g, is_clean, info):
    """cohort-specific confound path m_g(t) for t=1..T (1-indexed -> array idx t-1)."""
    t = np.arange(1, T + 1)
    if is_clean:
        return np.zeros(T)
    slope = info * V
    jump = (1 - info) * V
    return slope * (t - (g - 1)) + jump * (t >= g).astype(float)


def one_rep(info):
    # ----- generate panel & compute per-unit difference statistics -----
    # relevant periods per cohort: g-2, g-1, g   (indices g-3, g-2, g-1)
    # treated-unit stats
    tr_pre_mean = np.empty(G); tr_post_mean = np.empty(G)
    tr_cov = np.zeros((G, 2, 2))                 # within-cohort cov of (dpre,dpost)
    for k, g in enumerate(cohorts):
        m = m_path(g, clean_mask[k], info)
        b = g - 1
        # potential-outcome means at the three periods (alpha cancels in differences)
        # treated units: add theta at t>=g
        # sample eps for the three needed periods
        eps = rng.normal(0, sigma, size=(N_g, 3))      # cols: g-2, g-1, g
        y_gm2 = phi[g - 3] + m[g - 3] + eps[:, 0]
        y_gm1 = phi[g - 2] + m[g - 2] + eps[:, 1]
        y_g = phi[g - 1] + m[g - 1] + theta[k] + eps[:, 2]   # treated: +theta at t=g
        dpre = y_gm2 - y_gm1
        dpost = y_g - y_gm1
        tr_pre_mean[k] = dpre.mean(); tr_post_mean[k] = dpost.mean()
        tr_cov[k] = np.cov(np.vstack([dpre, dpost]))

    # control units (shared): compute their (dpre,dpost) for EVERY cohort
    # one set of control units, outcomes across all needed periods
    needed = sorted(set([g - 3 for g in cohorts] + [g - 2 for g in cohorts] + [g - 1 for g in cohorts]))
    epsc = {p: rng.normal(0, sigma, size=N_c) for p in range(T)}
    ctrl_stat = np.empty((N_c, 2 * G))           # [pre_1..pre_G, post_1..post_G]
    ctrl_pre_mean = np.empty(G); ctrl_post_mean = np.empty(G)
    for k, g in enumerate(cohorts):
        yc_gm2 = phi[g - 3] + epsc[g - 3]        # control: m=0, no treatment
        yc_gm1 = phi[g - 2] + epsc[g - 2]
        yc_g = phi[g - 1] + epsc[g - 1]
        dpre_c = yc_gm2 - yc_gm1
        dpost_c = yc_g - yc_gm1
        ctrl_stat[:, k] = dpre_c
        ctrl_stat[:, G + k] = dpost_c
        ctrl_pre_mean[k] = dpre_c.mean(); ctrl_post_mean[k] = dpost_c.mean()

    beta_pre = tr_pre_mean - ctrl_pre_mean
    beta_post = tr_post_mean - ctrl_post_mean

    # ----- estimated covariance of (beta_pre, beta_post) stacked as 2G vector -----
    Sig = np.zeros((2 * G, 2 * G))
    # control contribution (dense; couples all cohorts through shared units)
    Cc = np.cov(ctrl_stat, rowvar=False) / N_c
    Sig += Cc
    # treated contribution (block per cohort; independent across cohorts)
    for k in range(G):
        Sig[k, k] += tr_cov[k][0, 0] / N_g
        Sig[G + k, G + k] += tr_cov[k][1, 1] / N_g
        Sig[k, G + k] += tr_cov[k][0, 1] / N_g
        Sig[G + k, k] += tr_cov[k][1, 0] / N_g

    Sig_post = Sig[G:, G:]                        # GxG covariance among post coefficients

    # ----- selection + aggregation -----
    sel = np.abs(beta_pre) <= c
    if sel.sum() == 0:
        sel = np.ones(G, bool)

    cs_est = beta_post.mean()
    w_cs = np.ones(G) / G
    var_cs = w_cs @ Sig_post @ w_cs

    idx = np.where(sel)[0]
    latt_est = beta_post[idx].mean()
    w_la = np.zeros(G); w_la[idx] = 1.0 / len(idx)
    var_la = w_la @ Sig_post @ w_la              # uses cross-cohort cov among selected

    tgt = theta[idx].mean()                      # causal target for selected set = 1
    cov_latt = (latt_est - z*np.sqrt(var_la) <= tgt <= latt_est + z*np.sqrt(var_la))

    return dict(cs=cs_est, latt=latt_est, nsel=sel.sum(),
                dirty_in=np.sum(sel & ~clean_mask), cov=cov_latt,
                se_la=np.sqrt(var_la),
                # off-diagonal correlation among post coefs (shared-control realism check)
                mean_offdiag_corr=_mean_offdiag_corr(Sig_post))


def _mean_offdiag_corr(S):
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    off = C[~np.eye(len(S), dtype=bool)]
    return off.mean()


def run(info, n_reps, label=None):
    cs = np.empty(n_reps); la = np.empty(n_reps); nsel = np.empty(n_reps)
    din = np.empty(n_reps); cov = np.empty(n_reps); sel_tgt = np.empty(n_reps)
    corr = np.empty(n_reps); sela = np.empty(n_reps)
    for r in range(n_reps):
        o = one_rep(info)
        cs[r] = o['cs']; la[r] = o['latt']; nsel[r] = o['nsel']
        din[r] = o['dirty_in']; cov[r] = o['cov']; corr[r] = o['mean_offdiag_corr']
        sela[r] = o['se_la']
    res = dict(info=info, cs_mean=cs.mean(), cs_bias=cs.mean()-TRUE,
               la_mean=la.mean(), la_bias=la.mean()-TRUE, nsel=nsel.mean(),
               din=din.mean(), cov=cov.mean()*100, corr=corr.mean(),
               se_la=sela.mean(), la_sd=la.std())
    if label:
        print(f"\n===== {label} (info={info}) =====")
        print(f"  TRUE ATT = TRUE LATT = 1.0000")
        print(f"  CS   estimate: mean={res['cs_mean']:.4f}  bias={res['cs_bias']:+.4f}")
        print(f"  LATT estimate: mean={res['la_mean']:.4f}  bias={res['la_bias']:+.4f}")
        print(f"  avg selected = {res['nsel']:.2f}/{G}  (dirty included: {res['din']:.3f})")
        print(f"  est. aggregation SE (LATT): {res['se_la']:.4f}   vs  MC sd of LATT: {res['la_sd']:.4f}"
              f"   (Sigma_hat calibrated? these should match)")
        print(f"  mean off-diagonal corr among post coefs = {res['corr']:.3f}"
              f"   (shared-control cross-cohort correlation, real)")
        print(f"  naive 95% CI coverage of causal LATT (using Sigma_hat) = {res['cov']:.1f}%")
    return res


# ---- headline at the working regime ----
run(info=1.0, n_reps=3000, label="HEADLINE: panel micro-sim, working regime")

# ---- mini master-axis sweep to confirm the Tier 1 scope pattern replicates ----
print("\n\n===== Mini master-axis sweep (panel data, estimated Sigma_hat) =====")
print(f"  {'info':>5} | {'CS bias':>8} {'LATT bias':>9} | {'cover%':>7} | {'dirty in':>8} {'off-corr':>8}")
for info in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    r = run(info, n_reps=2000)
    print(f"  {info:5.2f} | {r['cs_bias']:+8.4f} {r['la_bias']:+9.4f} | "
          f"{r['cov']:7.1f} | {r['din']:8.3f} {r['corr']:8.3f}")

print("\nDone.")
