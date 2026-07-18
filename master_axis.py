"""
Master-axis sweep (Tier 1, reduced-form normal model)
-----------------------------------------------------
x-axis  = INFORMATIVENESS of pre-trends: how strongly a post-treatment
          parallel-trends violation reveals itself in the pre-trend.
          info high  -> confound visible -> selection excludes it -> survivors clean
          info = 0   -> confound invisible in pre -> passes selection -> survivors dirty

Design (theta = 1 for every cohort, so true ATT = true LATT = 1; any movement off
1 is bias, not effect heterogeneity):
    - N_clean "clean" cohorts:      delta_pre = 0,          delta_post = 0
    - N_conf  "confounded" cohorts: delta_pre = info * V,   delta_post = V  (V>0, directional)
Observed beta_hat_g ~ N( (delta_pre_g , 1 + delta_post_g), Sigma ),  Sigma has
within-cohort pre/post sampling correlation rho_noise.
Selection: keep cohort g iff |beta_hat_g,pre| <= c.
    CS   = mean beta_hat_post over ALL cohorts   (targets ATT = 1)
    LATT = mean beta_hat_post over SELECTED      (targets credible-subpop effect = 1)

We sweep `info` and record, at each value:
  * CS bias and LATT bias vs the truth (=1)
  * identification gap in the selected set (mean delta_post over selected)
  * causal coverage of naive and sample-split 95% CIs
  * average number of confounded cohorts that slip into the selected set (mechanism)
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

# ---- fixed DGP knobs ----
N_clean, N_conf = 6, 6
G = N_clean + N_conf
theta = np.ones(G)                 # true effect = 1 for all -> true ATT = true LATT = 1
V = 0.6                            # directional post-treatment violation of confounded cohorts
s = 0.20                          # sampling sd (pre and post)
rho_noise = 0.3                   # within-cohort pre/post sampling correlation
c = 0.40                          # selection threshold on |beta_hat_pre|
n_reps = 8000
z = norm.ppf(0.975)

cov = np.array([[s**2, rho_noise*s*s], [rho_noise*s*s, s**2]])
L = np.linalg.cholesky(cov)
clean_mask = np.array([True]*N_clean + [False]*N_conf)

info_grid = np.linspace(0.0, 1.5, 25)
TRUE = 1.0

cs_bias, latt_bias = [], []
id_gap = []
cov_naive, cov_split = [], []
conf_included = []

for info in info_grid:
    delta_pre = np.where(clean_mask, 0.0, info*V)
    delta_post = np.where(clean_mask, 0.0, V)
    beta_pre = delta_pre
    beta_post = theta + delta_post

    cs_e = np.empty(n_reps); la_e = np.empty(n_reps)
    gap = np.empty(n_reps)
    nc = np.empty(n_reps)
    cn = np.zeros(n_reps, bool); cs_ = np.zeros(n_reps, bool)

    for r in range(n_reps):
        bh = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T
        sel = np.abs(bh[:, 0]) <= c
        if sel.sum() == 0:
            sel = np.ones(G, bool)
        cs_e[r] = bh[:, 1].mean()
        la_e[r] = bh[:, 1][sel].mean()
        gap[r] = delta_post[sel].mean()               # identification gap in selected set
        nc[r] = np.sum(sel & ~clean_mask)
        # causal target for the realized selected set = mean true effect over selected = 1
        cau_tgt = theta[sel].mean()
        se = s/np.sqrt(sel.sum())
        cn[r] = (la_e[r]-z*se <= cau_tgt <= la_e[r]+z*se)
        # sample-splitting honest benchmark
        h1 = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T*np.sqrt(2)
        h2 = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T*np.sqrt(2)
        s1 = np.abs(h1[:, 0]) <= c
        if s1.sum() == 0:
            s1 = np.ones(G, bool)
        est2 = h2[:, 1][s1].mean(); se2 = (s*np.sqrt(2))/np.sqrt(s1.sum())
        cs_[r] = (est2-z*se2 <= theta[s1].mean() <= est2+z*se2)

    cs_bias.append(cs_e.mean()-TRUE)
    latt_bias.append(la_e.mean()-TRUE)
    id_gap.append(gap.mean())
    cov_naive.append(cn.mean()*100)
    cov_split.append(cs_.mean()*100)
    conf_included.append(nc.mean())

cs_bias = np.array(cs_bias); latt_bias = np.array(latt_bias)
id_gap = np.array(id_gap); cov_naive = np.array(cov_naive)
cov_split = np.array(cov_split); conf_included = np.array(conf_included)

# ------------- print a compact table -------------
print(f"{'info':>5} | {'CS bias':>8} {'LATT bias':>9} | {'id-gap':>7} | "
      f"{'naive%':>7} {'split%':>7} | {'#conf in':>8}")
for i in range(0, len(info_grid), 2):
    print(f"{info_grid[i]:5.2f} | {cs_bias[i]:+8.3f} {latt_bias[i]:+9.3f} | "
          f"{id_gap[i]:7.3f} | {cov_naive[i]:7.1f} {cov_split[i]:7.1f} | {conf_included[i]:8.2f}")

# ------------- figure -------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Master-axis sweep: everything is governed by pre-trend informativeness\n"
             "(theta=1 for all cohorts, so true ATT = true LATT = 1)", fontsize=12)

a = ax[0, 0]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, cs_bias, "o-", color="#c0392b", label="CS estimator (targets ATT)")
a.plot(info_grid, latt_bias, "s-", color="#2471a3", label="reweighted LATT (ours)")
a.set_title("Point-estimate bias vs truth (=1)")
a.set_xlabel("pre-trend informativeness  (info)")
a.set_ylabel("bias")
a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, id_gap, "d-", color="#7d3c98")
a.set_title("Identification gap in the selected set\n(mean delta_post among selected cohorts)")
a.set_xlabel("pre-trend informativeness  (info)")
a.set_ylabel("residual violation")

a = ax[1, 0]
a.axhline(95, color="green", lw=.9, ls="--", label="95% nominal")
a.plot(info_grid, cov_naive, "o-", color="#e67e22", label="naive CI")
a.plot(info_grid, cov_split, "s-", color="#16a085", label="sample-split CI")
a.set_title("Causal coverage of the selected LATT")
a.set_xlabel("pre-trend informativeness  (info)")
a.set_ylabel("coverage (%)")
a.set_ylim(60, 100)
a.legend(fontsize=8)

a = ax[1, 1]
a.plot(info_grid, conf_included, "^-", color="#555555")
a.set_title("Mechanism: confounded cohorts slipping into selection\n(out of 6 confounded)")
a.set_xlabel("pre-trend informativeness  (info)")
a.set_ylabel("avg # confounded selected")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "master_axis.png"
plt.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
