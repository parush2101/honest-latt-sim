"""
Master-axis sweep (reduced-form normal model, 4+4 event study)
--------------------------------------------------------------
x-axis = INFORMATIVENESS phi of pre-trends: how strongly a post-treatment
         parallel-trends violation reveals itself in the pre-period.
         phi high -> confound's curvature is foreshadowed pre -> screen excludes it -> survivors clean
         phi = 0  -> confound flat in pre, curved post -> passes the screen -> survivors dirty

Unified with the rest of the simulation section: a full event study (4 pre, 4
post, reference e=0), a curved differential trend, and the CURVATURE screen
(max |pre-period second difference|), the same functional the SD(M) FLCI bounds.

Design (theta = 1 for every cohort, so true ATT = true LATT = 1; any movement off
1 is bias, not effect heterogeneity):
    - N_clean clean cohorts:      delta = 0
    - N_conf  confounded cohorts: delta_e = (phi*C/2) e^2 for e<0, (C/2) e^2 for e>0
      -> pre-period curvature = phi*C (the screen signal), post-period curvature = C
Target aggregation = average post-treatment effect (l_post uniform).
    CS   = mean post-effect over ALL cohorts   (targets ATT = 1)
    LATT = mean post-effect over SELECTED      (targets credible-subpop effect = 1)

We sweep phi and record CS/LATT bias, the identification gap of the selected set,
causal coverage of naive and sample-split CIs, and the number of confounded
cohorts slipping through the screen.
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

rng = np.random.default_rng(2024)

pre_e, post_e = L2.pre_e, L2.post_e
npre, npost = L2.npre, L2.npost
l_post = np.ones(npost) / npost

# ---- fixed DGP knobs ----
N_clean, N_conf = 6, 6
G = N_clean + N_conf
theta = np.ones(G)                 # true effect = 1 for all -> true ATT = true LATT = 1
C = 0.35                           # confounded post-period curvature (matches inference-tier "dirty")
sigma = 0.07                       # per-coefficient sampling sd (curvature screen needs precision)
rho = 0.5                          # within-cohort AR(1) sampling correlation
c = 0.22                           # curvature-screen threshold
n_reps = 8000
z = norm.ppf(0.975)


def confound(Cc, phi):
    dpre = 0.5 * (phi * Cc) * (pre_e ** 2)
    dpost = 0.5 * Cc * (post_e ** 2)
    return np.concatenate([dpre, dpost])


def within_cov(sig, rr, n):
    idx = np.arange(n)
    R = rr ** np.abs(idx[:, None] - idx[None, :])
    return (sig ** 2) * R


Sig = within_cov(sigma, rho, npre + npost)
Lc = np.linalg.cholesky(Sig)
sigma_post1 = np.sqrt(l_post @ Sig[npre:, npre:] @ l_post)   # SE of one cohort's avg post effect
clean_mask = np.array([True] * N_clean + [False] * N_conf)
tau_block = np.concatenate([np.zeros(npre), np.ones(npost)])

info_grid = np.linspace(0.0, 1.5, 25)
TRUE = 1.0

cs_bias, latt_bias, id_gap, cov_naive, cov_split, conf_included = [], [], [], [], [], []

for phi in info_grid:
    means = np.array([
        tau_block + (confound(C, phi) if not clean_mask[g] else 0.0) for g in range(G)
    ])
    post_viol = np.array([
        (confound(C, phi)[npre:] @ l_post) if not clean_mask[g] else 0.0 for g in range(G)
    ])

    cs_e = np.empty(n_reps); la_e = np.empty(n_reps)
    gap = np.empty(n_reps); nc = np.empty(n_reps)
    cn = np.zeros(n_reps, bool); csp = np.zeros(n_reps, bool)

    for r in range(n_reps):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
        post_eff = B[:, npre:] @ l_post
        stat = L2.max_abs_second_diff(B[:, :npre])
        sel = stat <= c
        if sel.sum() == 0:
            sel = (stat == stat.min())
        cs_e[r] = post_eff.mean()
        la_e[r] = post_eff[sel].mean()
        gap[r] = post_viol[sel].mean()                       # identification gap in selected set
        nc[r] = np.sum(sel & ~clean_mask)
        se = sigma_post1 / np.sqrt(sel.sum())
        cn[r] = (la_e[r] - z*se <= theta[sel].mean() <= la_e[r] + z*se)
        # sample-splitting honest benchmark: select on h1, estimate on h2 (independent, sqrt(2) noise)
        B1 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
        B2 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
        stat1 = L2.max_abs_second_diff(B1[:, :npre])
        s1 = stat1 <= c
        if s1.sum() == 0:
            s1 = (stat1 == stat1.min())
        est2 = (B2[:, npre:] @ l_post)[s1].mean(); se2 = (sigma_post1 * np.sqrt(2)) / np.sqrt(s1.sum())
        csp[r] = (est2 - z*se2 <= theta[s1].mean() <= est2 + z*se2)

    cs_bias.append(cs_e.mean() - TRUE)
    latt_bias.append(la_e.mean() - TRUE)
    id_gap.append(gap.mean())
    cov_naive.append(cn.mean() * 100)
    cov_split.append(csp.mean() * 100)
    conf_included.append(nc.mean())

cs_bias = np.array(cs_bias); latt_bias = np.array(latt_bias)
id_gap = np.array(id_gap); cov_naive = np.array(cov_naive)
cov_split = np.array(cov_split); conf_included = np.array(conf_included)

# ------------- print a compact table -------------
print(f"{'phi':>5} | {'CS bias':>8} {'LATT bias':>9} | {'id-gap':>7} | "
      f"{'naive%':>7} {'split%':>7} | {'#conf in':>8}")
for i in range(0, len(info_grid), 2):
    print(f"{info_grid[i]:5.2f} | {cs_bias[i]:+8.3f} {latt_bias[i]:+9.3f} | "
          f"{id_gap[i]:7.3f} | {cov_naive[i]:7.1f} {cov_split[i]:7.1f} | {conf_included[i]:8.2f}")

# ------------- figure -------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Master-axis sweep: everything is governed by pre-trend informativeness\n"
             "(theta=1 for all cohorts, so true ATT = true LATT = 1; curvature screen)", fontsize=12)

a = ax[0, 0]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, cs_bias, "o-", color="#c0392b", label="CS estimator (targets ATT)")
a.plot(info_grid, latt_bias, "s-", color="#2471a3", label="reweighted LATT (ours)")
a.set_title("Point-estimate bias vs truth (=1)")
a.set_xlabel("pre-trend informativeness  (phi)")
a.set_ylabel("bias")
a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, id_gap, "d-", color="#7d3c98")
a.set_title("Identification gap in the selected set\n(mean post-treatment violation among selected)")
a.set_xlabel("pre-trend informativeness  (phi)")
a.set_ylabel("residual violation")

a = ax[1, 0]
a.axhline(95, color="green", lw=.9, ls="--", label="95% nominal")
a.plot(info_grid, cov_naive, "o-", color="#e67e22", label="naive CI")
a.plot(info_grid, cov_split, "s-", color="#16a085", label="sample-split CI")
a.set_title("Causal coverage of the selected LATT")
a.set_xlabel("pre-trend informativeness  (phi)")
a.set_ylabel("coverage (%)")
a.set_ylim(0, 100)
a.legend(fontsize=8)

a = ax[1, 1]
a.plot(info_grid, conf_included, "^-", color="#555555")
a.set_title("Mechanism: confounded cohorts slipping into selection\n(out of 6 confounded)")
a.set_xlabel("pre-trend informativeness  (phi)")
a.set_ylabel("avg # confounded selected")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "master_axis.png"
plt.savefig(out, dpi=130)
print(f"\nSaved figure -> {out}")
