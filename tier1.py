"""
Tier 1 simulation (reduced-form normal model, 4+4 event study)
--------------------------------------------------------------
The estimand demonstration. When some cohorts violate parallel trends,
  - the CS estimator aims at the full ATT and MISSES (biased point estimate),
  - our reweighted LATT aims at the credible-subpopulation effect and HITS it.

Unified with the inference tier: a full event study (4 pre, 4 post, reference
e=0), a curved differential trend, and the CURVATURE screen (max |pre-period
second difference|, via L2.max_abs_second_diff), the same functional the SD(M)
FLCI bounds. A confounded cohort has a curved trend with pre-period curvature
phi*C and post-period curvature C (phi = informativeness). At phi=1 the confound
is fully foreshadowed in the pre-period, so the screen can detect it.

Per cohort g, coefficients beta_g = tau_g + delta_g drawn ~ N(., Sigma) with a
within-cohort AR(1) covariance:
    tau_g   = (0_pre, 1_post)                 # no anticipation; true effect = 1
    delta_g = curved confound (clean: 0)
Target aggregation = average post-treatment effect (l_post uniform), so the true
ATT and true LATT are both 1 and any departure is a parallel-trends violation.

CS  point estimate  = mean post-effect over ALL cohorts      (targets ATT)
LATT point estimate = mean post-effect over SELECTED cohorts (targets LATT)

Note the curvature screen needs more per-cohort precision than the old level
screen (a second difference is a noisier functional), so the design uses a
tighter sigma; the sample-size sweep spans useless-screen to precise-screen.
"""

import numpy as np
import layer2_full as L2

rng = np.random.default_rng(12345)

pre_e, post_e = L2.pre_e, L2.post_e          # [-4..-1], [1..4]
npre, npost = L2.npre, L2.npost
l_post = np.ones(npost) / npost              # target: average post-treatment effect


def confound(C, phi):
    """Curved differential trend: pre curvature phi*C, post curvature C, delta_0=0."""
    dpre = 0.5 * (phi * C) * (pre_e ** 2)
    dpost = 0.5 * C * (post_e ** 2)
    return np.concatenate([dpre, dpost])


def within_cov(sigma, rho, n):
    idx = np.arange(n)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    return (sigma ** 2) * R


def run_scenario(theta, C, phi, sigma, rho, c, n_reps, label):
    """theta: per-cohort true effect; confounded cohorts have C>0."""
    G = len(theta)
    theta = np.asarray(theta, float)
    Cvec = np.asarray(C, float)
    clean_mask = (Cvec == 0.0)

    tau = np.concatenate([np.zeros(npre), np.ones(npost)])
    # per-cohort mean: tau scaled to the cohort's true effect on the post block, plus confound
    means = np.array([
        np.concatenate([np.zeros(npre), np.full(npost, theta[g])]) + confound(Cvec[g], phi)
        for g in range(G)
    ])
    # post-effect violation actually carried into the average-post aggregation
    post_viol = np.array([confound(Cvec[g], phi)[npre:] @ l_post for g in range(G)])

    true_ATT = theta.mean()
    true_LATT_clean = theta[clean_mask].mean()

    Sig = within_cov(sigma, rho, npre + npost)
    Lc = np.linalg.cholesky(Sig)

    cs_est = np.empty(n_reps)
    latt_est = np.empty(n_reps)
    n_sel = np.empty(n_reps, int)
    dirty_in = np.zeros(n_reps, int)
    clean_out = np.zeros(n_reps, int)

    for r in range(n_reps):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T   # (G, 8)
        post_eff = B[:, npre:] @ l_post                               # avg post effect per cohort
        stat = L2.max_abs_second_diff(B[:, :npre])                    # curvature screen statistic
        sel = stat <= c
        if sel.sum() == 0:
            sel = (stat == stat.min())                               # keep single most credible cohort
        cs_est[r] = post_eff.mean()
        latt_est[r] = post_eff[sel].mean()
        n_sel[r] = sel.sum()
        dirty_in[r] = np.sum(sel & ~clean_mask)
        clean_out[r] = np.sum(~sel & clean_mask)

    if label is not None:
        print(f"\n===== {label} =====")
        print(f"  cohorts: {G} ({clean_mask.sum()} clean, {(~clean_mask).sum()} confounded), "
              f"C={Cvec[~clean_mask][:1]}, phi={phi}, c={c}, sigma={sigma}, rho={rho}, reps={n_reps}")
        print(f"  post-effect violation per confounded cohort = {post_viol[~clean_mask][:1]}")
        print(f"  TRUE ATT (all)    = {true_ATT:.4f}   <- CS targets")
        print(f"  TRUE LATT (clean) = {true_LATT_clean:.4f}   <- we target")
        print("  ----------------------------------------------------------------")
        print(f"  CS  estimate: mean={cs_est.mean():.4f}  bias vs ATT ={cs_est.mean()-true_ATT:+.4f}")
        print(f"  LATT estimate: mean={latt_est.mean():.4f}  bias vs LATT={latt_est.mean()-true_LATT_clean:+.4f}")
        print("  ----------------------------------------------------------------")
        print(f"  avg # selected = {n_sel.mean():.2f}/{G}  "
              f"(confounded wrongly kept: {dirty_in.mean():.3f}, clean wrongly dropped: {clean_out.mean():.3f})")
    return dict(true_ATT=true_ATT, true_LATT=true_LATT_clean, cs=cs_est, latt=latt_est)


# ---------------- Scenario A: isolate the violation bias ----------------
# All cohorts share the SAME true effect (1). true ATT == true LATT == 1.
# CS moves off 1 only because of the parallel-trends VIOLATION in the confounded cohorts.
C_DIRTY = 0.35        # confounded post-curvature (matches inference-tier "dirty")
resA = run_scenario(
    theta=[1, 1, 1, 1, 1, 1],
    C=[0, 0, 0, 0, C_DIRTY, C_DIRTY], phi=1.0,
    sigma=0.07, rho=0.5, c=0.22, n_reps=20000,
    label="Scenario A - violation bias isolated (true ATT = true LATT = 1.0)")

# ---------------- Scenario B: estimand divergence ----------------
# Confounded cohorts ALSO have a different true effect (1.6): true ATT != true LATT.
resB = run_scenario(
    theta=[1, 1, 1, 1, 1.6, 1.6],
    C=[0, 0, 0, 0, C_DIRTY, C_DIRTY], phi=1.0,
    sigma=0.07, rho=0.5, c=0.22, n_reps=20000,
    label="Scenario B - estimand divergence (true ATT != true LATT)")


# ---------------- Sample-size sweep (Scenario A) ----------------
# Prediction: LATT bias -> 0 as precision grows (pre-test bias vanishes, screen sharpens),
#             while CS bias PERSISTS (wrong target, not a noise problem).
print("\n\n===== Sample-size sweep (Scenario A): LATT bias vanishes, CS bias persists? =====")
print(f"  {'sigma':>8} | {'CS mean':>8} {'CS bias':>8} | {'LATT mean':>9} {'LATT bias':>9} | {'avg#sel':>7}")
for s in [0.30, 0.20, 0.12, 0.07, 0.04, 0.02]:
    G = 6
    theta = np.ones(G)
    Cvec = np.array([0, 0, 0, 0, C_DIRTY, C_DIRTY], float)
    clean_mask = (Cvec == 0.0)
    means = np.array([
        np.concatenate([np.zeros(npre), np.full(npost, theta[g])]) + confound(Cvec[g], 1.0)
        for g in range(G)])
    Lc = np.linalg.cholesky(within_cov(s, 0.5, npre + npost))
    NR = 20000
    cs = np.empty(NR); la = np.empty(NR); nsel = np.empty(NR)
    for i in range(NR):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
        post_eff = B[:, npre:] @ l_post
        stat = L2.max_abs_second_diff(B[:, :npre])
        sel = stat <= 0.22
        if sel.sum() == 0: sel = (stat == stat.min())
        cs[i] = post_eff.mean(); la[i] = post_eff[sel].mean(); nsel[i] = sel.sum()
    print(f"  {s:8.2f} | {cs.mean():8.4f} {cs.mean()-1.0:+8.4f} | "
          f"{la.mean():9.4f} {la.mean()-1.0:+9.4f} | {nsel.mean():7.2f}")

print("\nDone.")
