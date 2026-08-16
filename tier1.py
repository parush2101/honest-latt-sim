"""
Tier 1 simulation (reduced-form normal model, 4+4 event study)
--------------------------------------------------------------
The estimand demonstration. When some cohorts violate parallel trends,
  - the CS estimator aims at the full ATT and MISSES (biased point estimate),
  - our reweighted LATT aims at the credible-subpopulation effect and HITS it.

Level/flatness world (the paper's main framework): parallel trends means a flat
treated-comparison difference, so a violation is a level shift of that difference.
A confounded cohort has a post-treatment level violation V, foreshadowed in the
pre-period as phi*V (phi = informativeness). The screen is the FLATNESS screen
(max |pre-coefficient| <= c), and honest inference is the level bound
Delta_Level(M)={|delta_post(e)|<=M} (developed in tier1b / the calibration cell).

Per cohort g, coefficients beta_g = tau_g + delta_g drawn ~ N(., Sigma) with a
within-cohort AR(1) covariance:
    tau_g   = (0_pre, 1_post)                 # no anticipation; true effect = 1
    delta_g = (phi*V on pre, V on post)       # clean: 0
Target aggregation = average post-treatment effect, so true ATT and true LATT are
both 1 and any departure is a parallel-trends violation.

CS  point estimate  = mean post-effect over ALL cohorts      (targets ATT)
LATT point estimate = mean post-effect over SELECTED cohorts (targets LATT)
"""

import numpy as np
import layer2_full as L2

rng = np.random.default_rng(12345)

pre_e, post_e = L2.pre_e, L2.post_e
npre, npost = L2.npre, L2.npost
l_post = np.ones(npost) / npost


def confound(V, phi):
    """Level violation: pre-period shift phi*V, post-period shift V, delta_0=0."""
    return np.concatenate([np.full(npre, phi * V), np.full(npost, V)])


def within_cov(sigma, rho, n):
    idx = np.arange(n)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    return (sigma ** 2) * R


def run_scenario(theta, V, phi, sigma, rho, c, n_reps, label):
    G = len(theta)
    theta = np.asarray(theta, float)
    Vvec = np.asarray(V, float)
    clean_mask = (Vvec == 0.0)

    means = np.array([
        np.concatenate([np.zeros(npre), np.full(npost, theta[g])]) + confound(Vvec[g], phi)
        for g in range(G)
    ])
    true_ATT = theta.mean()
    true_LATT_clean = theta[clean_mask].mean()

    Sig = within_cov(sigma, rho, npre + npost)
    Lc = np.linalg.cholesky(Sig)

    cs_est = np.empty(n_reps); latt_est = np.empty(n_reps)
    n_sel = np.empty(n_reps, int); dirty_in = np.zeros(n_reps, int); clean_out = np.zeros(n_reps, int)

    for r in range(n_reps):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
        post_eff = B[:, npre:] @ l_post
        stat = np.max(np.abs(B[:, :npre]), axis=1)          # FLATNESS screen (magnitude)
        sel = stat <= c
        if sel.sum() == 0:
            sel = (stat == stat.min())
        cs_est[r] = post_eff.mean()
        latt_est[r] = post_eff[sel].mean()
        n_sel[r] = sel.sum()
        dirty_in[r] = np.sum(sel & ~clean_mask); clean_out[r] = np.sum(~sel & clean_mask)

    if label is not None:
        print(f"\n===== {label} =====")
        print(f"  cohorts: {G} ({clean_mask.sum()} clean, {(~clean_mask).sum()} confounded), "
              f"V={Vvec[~clean_mask][:1]}, phi={phi}, c={c}, sigma={sigma}, rho={rho}, reps={n_reps}")
        print(f"  TRUE ATT (all)    = {true_ATT:.4f}   <- CS targets")
        print(f"  TRUE LATT (clean) = {true_LATT_clean:.4f}   <- we target")
        print("  ----------------------------------------------------------------")
        print(f"  CS  estimate: mean={cs_est.mean():.4f}  bias vs ATT ={cs_est.mean()-true_ATT:+.4f}")
        print(f"  LATT estimate: mean={latt_est.mean():.4f}  bias vs LATT={latt_est.mean()-true_LATT_clean:+.4f}")
        print("  ----------------------------------------------------------------")
        print(f"  avg # selected = {n_sel.mean():.2f}/{G}  "
              f"(confounded kept: {dirty_in.mean():.3f}, clean dropped: {clean_out.mean():.3f})")
    return dict(true_ATT=true_ATT, true_LATT=true_LATT_clean, cs=cs_est, latt=latt_est)


# ---------------- Scenario A: isolate the violation bias ----------------
V_DIRTY = 0.8
resA = run_scenario(
    theta=[1, 1, 1, 1, 1, 1],
    V=[0, 0, 0, 0, V_DIRTY, V_DIRTY], phi=1.0,
    sigma=0.25, rho=0.5, c=0.40, n_reps=20000,
    label="Scenario A - violation bias isolated (true ATT = true LATT = 1.0)")

# ---------------- Scenario B: estimand divergence ----------------
resB = run_scenario(
    theta=[1, 1, 1, 1, 1.6, 1.6],
    V=[0, 0, 0, 0, V_DIRTY, V_DIRTY], phi=1.0,
    sigma=0.25, rho=0.5, c=0.40, n_reps=20000,
    label="Scenario B - estimand divergence (true ATT != true LATT)")


# ---------------- Sample-size sweep (Scenario A) ----------------
print("\n\n===== Sample-size sweep (Scenario A): LATT bias vanishes, CS bias persists? =====")
print(f"  {'sigma':>8} | {'CS mean':>8} {'CS bias':>8} | {'LATT mean':>9} {'LATT bias':>9} | {'avg#sel':>7}")
for s in [0.40, 0.30, 0.20, 0.12, 0.07, 0.03]:
    G = 6
    theta = np.ones(G)
    Vvec = np.array([0, 0, 0, 0, V_DIRTY, V_DIRTY], float)
    means = np.array([
        np.concatenate([np.zeros(npre), np.full(npost, 1.0)]) + confound(Vvec[g], 1.0)
        for g in range(G)])
    Lc = np.linalg.cholesky(within_cov(s, 0.5, npre + npost))
    NR = 20000
    cs = np.empty(NR); la = np.empty(NR); nsel = np.empty(NR)
    for i in range(NR):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
        post_eff = B[:, npre:] @ l_post
        stat = np.max(np.abs(B[:, :npre]), axis=1)
        sel = stat <= 0.40
        if sel.sum() == 0: sel = (stat == stat.min())
        cs[i] = post_eff.mean(); la[i] = post_eff[sel].mean(); nsel[i] = sel.sum()
    print(f"  {s:8.2f} | {cs.mean():8.4f} {cs.mean()-1.0:+8.4f} | "
          f"{la.mean():9.4f} {la.mean()-1.0:+9.4f} | {nsel.mean():7.2f}")

print("\nDone.")
