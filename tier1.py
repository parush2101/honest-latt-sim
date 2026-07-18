"""
Tier 1 simulation (clean-lab / reduced-form normal model)
----------------------------------------------------------
Question the paper makes: when some cohorts violate parallel trends,
  - the CS estimator aims at the full ATT and MISSES (biased point estimate),
  - our reweighted LATT aims at the credible-subpopulation effect and HITS it.

We build a known world (so the truth is known), draw the event-study
coefficients directly from a normal model, and compare the two POINT ESTIMATES
against the two true targets. Interval behaviour is a secondary panel.

Per cohort g we use a 2-vector (one summary pre coefficient, one summary post
coefficient), the RR "three-period" reduced form:
    beta_g = tau_g + delta_g
    tau_g  = (0, theta_g)                      # no anticipation; theta_g = true effect
    delta_g= (d_g, link*d_g)                    # pre & post violation (strong link)
Clean cohort: d_g = 0  -> beta_g = (0, theta_g)
Dirty cohort: d_g > 0  -> beta_g = (d_g, theta_g + link*d_g)

Observed: beta_hat_g ~ N(beta_g, Sigma_g), Sigma_g has pre/post correlation rho.
Selection rule (polyhedral box): include cohort g iff |beta_hat_g,pre| <= c.
CS  point estimate  = mean of beta_hat_g,post over ALL cohorts      (targets ATT)
LATT point estimate = mean of beta_hat_g,post over SELECTED cohorts (targets LATT)
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(12345)


def run_scenario(theta, d, link, s_pre, s_post, rho, c, n_reps, label):
    G = len(theta)
    theta = np.asarray(theta, float)
    d = np.asarray(d, float)
    clean_mask = (d == 0.0)

    # True reduced-form means
    beta_pre = d.copy()                     # tau_pre = 0
    beta_post = theta + link * d            # tau_post + delta_post
    delta_post = link * d

    # --- true targets (known because we built the world) ---
    true_ATT = theta.mean()                         # avg causal effect over ALL cohorts
    true_LATT_clean = theta[clean_mask].mean()      # avg causal effect over CLEAN cohorts

    # per-cohort 2x2 covariance (pre, post) with correlation rho
    cov = np.array([[s_pre**2, rho * s_pre * s_post],
                    [rho * s_pre * s_post, s_post**2]])
    L = np.linalg.cholesky(cov)

    cs_est = np.empty(n_reps)
    latt_est = np.empty(n_reps)
    n_selected = np.empty(n_reps, int)
    sel_LATT_true = np.empty(n_reps)        # realized selected causal target per rep
    naive_cover = np.zeros(n_reps, bool)    # naive CI coverage of selected LATT
    dirty_included = np.zeros(n_reps, int)
    clean_excluded = np.zeros(n_reps, int)

    z = norm.ppf(0.975)
    for r in range(n_reps):
        noise = (L @ rng.standard_normal((2, G))).T          # (G,2)
        bhat = np.column_stack([beta_pre, beta_post]) + noise
        bpre, bpost = bhat[:, 0], bhat[:, 1]

        sel = np.abs(bpre) <= c
        if sel.sum() == 0:
            sel = np.ones(G, bool)                            # degenerate guard

        cs_est[r] = bpost.mean()
        latt_est[r] = bpost[sel].mean()
        n_selected[r] = sel.sum()
        # realized selected causal target = mean true effect over selected cohorts
        sel_LATT_true[r] = theta[sel].mean()
        dirty_included[r] = np.sum(sel & ~clean_mask)
        clean_excluded[r] = np.sum(~sel & clean_mask)

        # NAIVE CI for the selected LATT (ignores that selection happened)
        se_naive = s_post / np.sqrt(sel.sum())
        lo, hi = latt_est[r] - z * se_naive, latt_est[r] + z * se_naive
        naive_cover[r] = (lo <= sel_LATT_true[r] <= hi)

    print(f"\n===== {label} =====")
    print(f"  cohorts: {G}  ({clean_mask.sum()} clean, {(~clean_mask).sum()} dirty),"
          f"  link={link}, c={c}, s_post={s_post}, rho={rho}, reps={n_reps}")
    print(f"  TRUE  ATT (all cohorts)   = {true_ATT:.4f}   <- what CS targets")
    print(f"  TRUE  LATT (clean cohorts)= {true_LATT_clean:.4f}   <- what WE target")
    print("  ----------------------------------------------------------------")
    print(f"  CS  estimate:  mean={cs_est.mean():.4f}  bias vs true ATT ={cs_est.mean()-true_ATT:+.4f}")
    print(f"                                 bias vs true LATT={cs_est.mean()-true_LATT_clean:+.4f}")
    print(f"  LATT estimate: mean={latt_est.mean():.4f}  bias vs true LATT={latt_est.mean()-true_LATT_clean:+.4f}")
    print(f"                                 bias vs true ATT ={latt_est.mean()-true_ATT:+.4f}")
    print("  ----------------------------------------------------------------")
    print(f"  avg # selected = {n_selected.mean():.2f} / {G}"
          f"   (dirty wrongly included: {dirty_included.mean():.3f},"
          f"  clean wrongly excluded: {clean_excluded.mean():.3f})")
    print(f"  NAIVE 95% CI coverage of the (realized) selected LATT = {naive_cover.mean()*100:.1f}%"
          f"   (should be ~95% if honest)")
    return dict(true_ATT=true_ATT, true_LATT=true_LATT_clean,
                cs=cs_est, latt=latt_est)


# ---------------- Scenario A: isolate the violation bias ----------------
# All cohorts share the SAME true effect (theta=1). So true ATT == true LATT == 1.
# The ONLY reason CS moves off 1 is the parallel-trends VIOLATION in the dirty cohorts.
resA = run_scenario(
    theta=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    d    =[0.0, 0.0, 0.0, 0.0, 0.8, 0.8],   # last two dirty
    link=1.0, s_pre=0.25, s_post=0.25, rho=0.5, c=0.4,
    n_reps=20000, label="Scenario A - violation bias isolated (true ATT = true LATT = 1.0)")

# ---------------- Scenario B: estimand divergence ----------------
# Dirty cohorts ALSO have a different true effect (1.6). Now true ATT != true LATT,
# AND CS is further corrupted by the violation. Shows the full estimand-choice story.
resB = run_scenario(
    theta=[1.0, 1.0, 1.0, 1.0, 1.6, 1.6],
    d    =[0.0, 0.0, 0.0, 0.0, 0.8, 0.8],
    link=1.0, s_pre=0.25, s_post=0.25, rho=0.5, c=0.4,
    n_reps=20000, label="Scenario B - estimand divergence (true ATT != true LATT)")


# ---------------- Sample-size sweep (Scenario A) ----------------
# Prediction: LATT bias -> 0 as precision grows (pre-test bias vanishes),
#             while CS bias PERSISTS (wrong target, not a noise problem).
print("\n\n===== Sample-size sweep (Scenario A): does LATT bias vanish, CS bias persist? =====")
print(f"  {'s_post=s_pre':>12} | {'CS mean':>8} {'CS bias':>8} | {'LATT mean':>9} {'LATT bias':>9}")
for s in [0.40, 0.30, 0.20, 0.12, 0.07, 0.03]:
    r = run_scenario(theta=[1,1,1,1,1,1], d=[0,0,0,0,0.8,0.8], link=1.0,
                     s_pre=s, s_post=s, rho=0.5, c=0.4, n_reps=20000,
                     label=None) if False else None
    # inline compute to avoid huge printout
    G = 6
    theta = np.ones(G); d = np.array([0,0,0,0,0.8,0.8]); link = 1.0
    clean = d == 0
    beta_pre = d; beta_post = theta + link*d
    cov = np.array([[s**2, 0.5*s*s],[0.5*s*s, s**2]]); L = np.linalg.cholesky(cov)
    NR = 20000
    cs = np.empty(NR); la = np.empty(NR)
    for i in range(NR):
        noise = (L @ rng.standard_normal((2,G))).T
        bh = np.column_stack([beta_pre, beta_post]) + noise
        sel = np.abs(bh[:,0]) <= 0.4
        if sel.sum()==0: sel = np.ones(G,bool)
        cs[i] = bh[:,1].mean(); la[i] = bh[:,1][sel].mean()
    true_ATT = 1.0; true_LATT = 1.0
    print(f"  {s:12.2f} | {cs.mean():8.4f} {cs.mean()-true_ATT:+8.4f} | "
          f"{la.mean():9.4f} {la.mean()-true_LATT:+9.4f}")

print("\nDone.")
