"""
Tier 2 simulation: full staggered PANEL micro-data + real CS-style estimation
-----------------------------------------------------------------------------
Panel confirmation of the reduced-form results, now on the 4+4 curvature world.
We generate individual panel data, estimate the cohort-level 4+4 event-study
coefficients with actual difference-in-differences contrasts against a shared
never-treated control, estimate their covariance FROM THE DATA, and apply the
CURVATURE screen (max |pre-period second difference|). A shared control induces
real cross-cohort correlation the aggregation standard error must account for.

DGP (untreated potential outcome; unit and time effects cancel in the DiD contrast):
    Y_it(0) = m_g(t) + eps_it,   eps ~ N(0, sigma^2)
  clean cohort:      m_g = 0
  confounded cohort: differential trend curved on event time e = t-(g-1):
        delta_g(e) = (phi*C/2) e^2  for e<0,   (C/2) e^2  for e>0
      -> pre-period curvature phi*C (the screen signal), post-period curvature C
Treatment: Y_it(1) = Y_it(0) + theta_g 1{t>=g}, theta_g = 1 (bias isolation).

Event-study coefficient (vs never-treated control, base period b=g-1):
    beta_g(e) = [Ybar_g(b+e) - Ybar_g(b)] - [Ybar_c(b+e) - Ybar_c(b)]
              = delta_g(e) + theta 1{e>=1} + noise.
Sigma_hat for the post-effect aggregate: treated part block-diagonal (independent
units per cohort); shared-control part dense (couples cohorts).

Selection: curvature screen on the estimated pre-trend.
CS = mean post-effect over ALL cohorts; LATT = mean over SELECTED.
"""

import numpy as np
from scipy.stats import norm
import layer2_full as L2

rng = np.random.default_rng(20240718)

# ---- design ----
cohorts = np.array([6, 7, 8, 9, 10, 11])          # each has 4 pre + 4 post periods within T
clean_mask = np.array([True, True, True, False, False, False])
G = len(cohorts)
T = 14
N_g = 500                                          # units per treated cohort
N_c = 1500                                         # shared never-treated control units
C = 0.35                                            # confounded post-period curvature
theta = 1.0
sigma = 0.5                                          # idiosyncratic noise sd
c_sel = 0.20                                         # curvature-screen threshold
z = norm.ppf(0.975)
TRUE = 1.0

pre_e = np.array([-4, -3, -2, -1])
post_e = np.array([1, 2, 3, 4])
ev = np.concatenate([pre_e, post_e])                # 8 event times
l_post = np.ones(4) / 4


def delta_curve(e, phi):
    e = np.asarray(e, float)
    return np.where(e < 0, 0.5 * (phi * C) * e ** 2, 0.5 * C * e ** 2)


def one_rep(phi):
    Yc = rng.normal(0, sigma, size=(N_c, T))        # control panel (period idx 0..T-1 = periods 1..T)
    beta = np.zeros((G, 8))
    pe_tr_var = np.zeros(G)
    pe_ctrl = np.zeros((N_c, G))                     # per-control-unit post-effect for each cohort
    for k, g in enumerate(cohorts):
        base = g - 1
        tper = base + ev                            # event-time periods (1-indexed)
        Yt = rng.normal(0, sigma, size=(N_g, T))
        if not clean_mask[k]:                       # add confound at every period
            for p in range(1, T + 1):
                Yt[:, p - 1] += float(delta_curve(p - base, phi))
        for p in range(g, T + 1):                   # treatment effect for t>=g
            Yt[:, p - 1] += theta
        d_tr = Yt[:, tper - 1] - Yt[:, base - 1][:, None]     # (N_g, 8)
        d_ct = Yc[:, tper - 1] - Yc[:, base - 1][:, None]     # (N_c, 8)
        beta[k] = d_tr.mean(0) - d_ct.mean(0)
        pe_tr = d_tr[:, 4:] @ l_post
        pe_tr_var[k] = pe_tr.var(ddof=1) / N_g
        pe_ctrl[:, k] = d_ct[:, 4:] @ l_post

    post_eff = beta[:, 4:] @ l_post                  # per-cohort post-effect estimate
    Sig_pe = np.diag(pe_tr_var) + np.cov(pe_ctrl, rowvar=False) / N_c   # treated block + shared-control dense

    stat = L2.max_abs_second_diff(beta[:, :4])       # curvature screen on estimated pre-trend
    sel = stat <= c_sel
    if sel.sum() == 0:
        sel = (stat == stat.min())

    cs_est = post_eff.mean()
    idx = np.where(sel)[0]
    latt_est = post_eff[idx].mean()
    w = np.zeros(G); w[idx] = 1.0 / len(idx)
    se_la = np.sqrt(w @ Sig_pe @ w)
    cov = (latt_est - z * se_la <= TRUE <= latt_est + z * se_la)
    d = np.sqrt(np.diag(Sig_pe)); corr = (Sig_pe / np.outer(d, d))[~np.eye(G, dtype=bool)].mean()
    return dict(cs=cs_est, latt=latt_est, nsel=sel.sum(),
                dirty_in=np.sum(sel & ~clean_mask), se_la=se_la, cov=cov, corr=corr)


def run(phi, n_reps, label=None):
    cs = np.empty(n_reps); la = np.empty(n_reps); nsel = np.empty(n_reps)
    din = np.empty(n_reps); sela = np.empty(n_reps); cov = np.empty(n_reps); corr = np.empty(n_reps)
    for r in range(n_reps):
        o = one_rep(phi)
        cs[r] = o['cs']; la[r] = o['latt']; nsel[r] = o['nsel']; din[r] = o['dirty_in']
        sela[r] = o['se_la']; cov[r] = o['cov']; corr[r] = o['corr']
    res = dict(phi=phi, cs_bias=cs.mean() - TRUE, la_bias=la.mean() - TRUE, nsel=nsel.mean(),
               din=din.mean(), se_la=sela.mean(), la_sd=la.std(), cov=cov.mean() * 100, corr=corr.mean())
    if label:
        print(f"\n===== {label} (phi={phi}) =====")
        print(f"  TRUE ATT = TRUE LATT = 1.0000")
        print(f"  CS   bias = {res['cs_bias']:+.4f}     LATT bias = {res['la_bias']:+.4f}")
        print(f"  avg selected = {res['nsel']:.2f}/{G}  (confounded included: {res['din']:.3f})")
        print(f"  est. aggregation SE (LATT) = {res['se_la']:.4f}   vs   MC sd of LATT = {res['la_sd']:.4f}")
        print(f"  mean off-diagonal corr among post-effects = {res['corr']:.3f}  (shared-control coupling)")
        print(f"  naive 95% CI coverage of causal LATT (using Sigma_hat) = {res['cov']:.1f}%")
    return res


run(phi=1.0, n_reps=3000, label="HEADLINE: panel micro-sim, fully-informative regime")

print("\n\n===== Mini master-axis sweep (panel data, estimated Sigma_hat, curvature screen) =====")
print(f"  {'phi':>5} | {'CS bias':>8} {'LATT bias':>9} | {'cover%':>7} | {'se_est':>7} {'mc_sd':>7} | {'dirty in':>8}")
for phi in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    r = run(phi, n_reps=2000)
    print(f"  {phi:5.2f} | {r['cs_bias']:+8.4f} {r['la_bias']:+9.4f} | {r['cov']:7.1f} | "
          f"{r['se_la']:7.4f} {r['la_sd']:7.4f} | {r['din']:8.3f}")

print("\nDone.")
