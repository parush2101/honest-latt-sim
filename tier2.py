"""
Tier 2 simulation: full staggered PANEL micro-data + real CS-style estimation
-----------------------------------------------------------------------------
Panel confirmation on the level/flatness world (the paper's main framework). We
generate individual panel data, estimate cohort-level 4+4 event-study coefficients
via difference-in-differences contrasts against a shared never-treated control,
estimate their covariance FROM THE DATA, and apply the FLATNESS screen
(max |pre-coefficient| <= c). A shared control induces cross-cohort correlation.

DGP (untreated potential outcome; unit and time effects cancel in the DiD contrast):
    Y_it(0) = delta_g(e) + eps_it,  e = t-(g-1),  eps ~ N(0, sigma^2)
  clean cohort:      delta_g = 0
  confounded cohort: level violation, delta_g(e) = phi*V for e<0, V for e>0, 0 at e=0
      -> pre-period shift phi*V (the screen signal), post-period shift V
Treatment: Y_it(1) = Y_it(0) + theta_g 1{t>=g}, theta_g = 1 (bias isolation).

CS = mean post-effect over ALL cohorts; LATT = mean over SELECTED.
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20240718)

cohorts = np.array([6, 7, 8, 9, 10, 11])
clean_mask = np.array([True, True, True, False, False, False])
G = len(cohorts)
T = 14
N_g = 500
N_c = 1500
V = 0.6                                              # confounded post-treatment level violation
theta = 1.0
sigma = 0.5
c_sel = 0.40                                          # flatness-screen threshold
z = norm.ppf(0.975)
TRUE = 1.0

pre_e = np.array([-4, -3, -2, -1])
post_e = np.array([1, 2, 3, 4])
ev = np.concatenate([pre_e, post_e])
l_post = np.ones(4) / 4


def delta_level(e, phi):
    e = np.asarray(e, float)
    return np.where(e < 0, phi * V, np.where(e > 0, V, 0.0))


def one_rep(phi):
    Yc = rng.normal(0, sigma, size=(N_c, T))
    beta = np.zeros((G, 8))
    pe_tr_var = np.zeros(G)
    pe_ctrl = np.zeros((N_c, G))
    for k, g in enumerate(cohorts):
        base = g - 1
        tper = base + ev
        Yt = rng.normal(0, sigma, size=(N_g, T))
        if not clean_mask[k]:
            for p in range(1, T + 1):
                Yt[:, p - 1] += float(delta_level(p - base, phi))
        for p in range(g, T + 1):
            Yt[:, p - 1] += theta
        d_tr = Yt[:, tper - 1] - Yt[:, base - 1][:, None]
        d_ct = Yc[:, tper - 1] - Yc[:, base - 1][:, None]
        beta[k] = d_tr.mean(0) - d_ct.mean(0)
        pe_tr = d_tr[:, 4:] @ l_post
        pe_tr_var[k] = pe_tr.var(ddof=1) / N_g
        pe_ctrl[:, k] = d_ct[:, 4:] @ l_post

    post_eff = beta[:, 4:] @ l_post
    Sig_pe = np.diag(pe_tr_var) + np.cov(pe_ctrl, rowvar=False) / N_c

    stat = np.max(np.abs(beta[:, :4]), axis=1)         # FLATNESS screen
    sel = stat <= c_sel
    if sel.sum() == 0:
        sel = (stat == stat.min())

    cs_est = post_eff.mean()
    idx = np.where(sel)[0]
    latt_est = post_eff[idx].mean()
    w = np.zeros(G); w[idx] = 1.0 / len(idx)
    se_la = np.sqrt(w @ Sig_pe @ w)
    cov = (latt_est - z * se_la <= TRUE <= latt_est + z * se_la)
    return dict(cs=cs_est, latt=latt_est, nsel=sel.sum(),
                dirty_in=np.sum(sel & ~clean_mask), se_la=se_la, cov=cov)


def run(phi, n_reps, label=None):
    cs = np.empty(n_reps); la = np.empty(n_reps); nsel = np.empty(n_reps)
    din = np.empty(n_reps); sela = np.empty(n_reps); cov = np.empty(n_reps)
    for r in range(n_reps):
        o = one_rep(phi)
        cs[r] = o['cs']; la[r] = o['latt']; nsel[r] = o['nsel']; din[r] = o['dirty_in']
        sela[r] = o['se_la']; cov[r] = o['cov']
    res = dict(phi=phi, cs_bias=cs.mean() - TRUE, la_bias=la.mean() - TRUE, nsel=nsel.mean(),
               din=din.mean(), se_la=sela.mean(), la_sd=la.std(), cov=cov.mean() * 100)
    if label:
        print(f"\n===== {label} (phi={phi}) =====")
        print(f"  TRUE ATT = TRUE LATT = 1.0000")
        print(f"  CS   bias = {res['cs_bias']:+.4f}     LATT bias = {res['la_bias']:+.4f}")
        print(f"  avg selected = {res['nsel']:.2f}/{G}  (confounded included: {res['din']:.3f})")
        print(f"  est. aggregation SE (LATT) = {res['se_la']:.4f}   vs   MC sd of LATT = {res['la_sd']:.4f}")
        print(f"  naive 95% CI coverage of causal LATT (using Sigma_hat) = {res['cov']:.1f}%")
    return res


run(phi=1.0, n_reps=3000, label="HEADLINE: panel micro-sim, fully-informative regime")

print("\n\n===== Mini master-axis sweep (panel data, estimated Sigma_hat, flatness screen) =====")
print(f"  {'phi':>5} | {'CS bias':>8} {'LATT bias':>9} | {'cover%':>7} | {'se_est':>7} {'mc_sd':>7} | {'dirty in':>8}")
for phi in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    r = run(phi, n_reps=2000)
    print(f"  {phi:5.2f} | {r['cs_bias']:+8.4f} {r['la_bias']:+9.4f} | {r['cov']:7.1f} | "
          f"{r['se_la']:7.4f} {r['la_sd']:7.4f} | {r['din']:8.3f}")

print("\nDone.")
