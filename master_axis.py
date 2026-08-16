"""
Master-axis sweep (reduced-form normal model, 4+4 event study)
--------------------------------------------------------------
x-axis = INFORMATIVENESS phi of pre-trends: how strongly a post-treatment
         parallel-trends violation reveals itself in the pre-period.
         phi high -> confound's level shift is foreshadowed pre -> screen excludes it -> survivors clean
         phi = 0  -> confound flat in pre, shifted post -> passes the screen -> survivors dirty

Level/flatness world (the paper's main framework). A confounded cohort has a
post-treatment level violation V and a pre-period foreshadow phi*V; the FLATNESS
screen (max |pre-coefficient| <= c) keeps cohorts whose pre-difference is near flat.

Design (theta = 1 for every cohort, so true ATT = true LATT = 1):
    - N_clean clean cohorts:      delta = 0
    - N_conf  confounded cohorts: delta_pre = phi*V, delta_post = V
Target aggregation = average post-treatment effect.
    CS   = mean post-effect over ALL cohorts   (targets ATT = 1)
    LATT = mean post-effect over SELECTED      (targets credible-subpop effect = 1)
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

N_clean, N_conf = 6, 6
G = N_clean + N_conf
theta = np.ones(G)
V = 0.6                            # post-treatment level violation of confounded cohorts
sigma = 0.20
rho = 0.5
c = 0.40                           # flatness-screen threshold
n_reps = 8000
z = norm.ppf(0.975)


def confound(Vv, phi):
    return np.concatenate([np.full(npre, phi * Vv), np.full(npost, Vv)])


def within_cov(sig, rr, n):
    idx = np.arange(n)
    R = rr ** np.abs(idx[:, None] - idx[None, :])
    return (sig ** 2) * R


Sig = within_cov(sigma, rho, npre + npost)
Lc = np.linalg.cholesky(Sig)
sigma_post1 = np.sqrt(l_post @ Sig[npre:, npre:] @ l_post)
clean_mask = np.array([True] * N_clean + [False] * N_conf)
tau_block = np.concatenate([np.zeros(npre), np.ones(npost)])

info_grid = np.linspace(0.0, 1.5, 25)
TRUE = 1.0

cs_bias, latt_bias, id_gap, cov_naive, cov_split, conf_included = [], [], [], [], [], []

for phi in info_grid:
    means = np.array([tau_block + (confound(V, phi) if not clean_mask[g] else 0.0) for g in range(G)])
    post_viol = np.array([(confound(V, phi)[npre:] @ l_post) if not clean_mask[g] else 0.0 for g in range(G)])

    cs_e = np.empty(n_reps); la_e = np.empty(n_reps)
    gap = np.empty(n_reps); nc = np.empty(n_reps)
    cn = np.zeros(n_reps, bool); csp = np.zeros(n_reps, bool)

    for r in range(n_reps):
        B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
        post_eff = B[:, npre:] @ l_post
        stat = np.max(np.abs(B[:, :npre]), axis=1)               # FLATNESS screen
        sel = stat <= c
        if sel.sum() == 0:
            sel = (stat == stat.min())
        cs_e[r] = post_eff.mean()
        la_e[r] = post_eff[sel].mean()
        gap[r] = post_viol[sel].mean()
        nc[r] = np.sum(sel & ~clean_mask)
        se = sigma_post1 / np.sqrt(sel.sum())
        cn[r] = (la_e[r] - z*se <= theta[sel].mean() <= la_e[r] + z*se)
        B1 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
        B2 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
        s1 = np.max(np.abs(B1[:, :npre]), axis=1) <= c
        if s1.sum() == 0:
            st = np.max(np.abs(B1[:, :npre]), axis=1); s1 = (st == st.min())
        est2 = (B2[:, npre:] @ l_post)[s1].mean(); se2 = (sigma_post1 * np.sqrt(2)) / np.sqrt(s1.sum())
        csp[r] = (est2 - z*se2 <= theta[s1].mean() <= est2 + z*se2)

    cs_bias.append(cs_e.mean() - TRUE); latt_bias.append(la_e.mean() - TRUE)
    id_gap.append(gap.mean()); cov_naive.append(cn.mean() * 100)
    cov_split.append(csp.mean() * 100); conf_included.append(nc.mean())

cs_bias = np.array(cs_bias); latt_bias = np.array(latt_bias)
id_gap = np.array(id_gap); cov_naive = np.array(cov_naive)
cov_split = np.array(cov_split); conf_included = np.array(conf_included)

print(f"{'phi':>5} | {'CS bias':>8} {'LATT bias':>9} | {'id-gap':>7} | {'naive%':>7} {'split%':>7} | {'#conf in':>8}")
for i in range(0, len(info_grid), 2):
    print(f"{info_grid[i]:5.2f} | {cs_bias[i]:+8.3f} {latt_bias[i]:+9.3f} | "
          f"{id_gap[i]:7.3f} | {cov_naive[i]:7.1f} {cov_split[i]:7.1f} | {conf_included[i]:8.2f}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Master-axis sweep: everything is governed by pre-trend informativeness\n"
             "(theta=1 for all cohorts, so true ATT = true LATT = 1; flatness screen)", fontsize=12)

a = ax[0, 0]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, cs_bias, "o-", color="#c0392b", label="CS estimator (targets ATT)")
a.plot(info_grid, latt_bias, "s-", color="#2471a3", label="reweighted LATT (ours)")
a.set_title("Point-estimate bias vs truth (=1)")
a.set_xlabel("pre-trend informativeness  (phi)"); a.set_ylabel("bias"); a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(0, color="gray", lw=.8, ls=":")
a.plot(info_grid, id_gap, "d-", color="#7d3c98")
a.set_title("Identification gap in the selected set\n(mean post-treatment violation among selected)")
a.set_xlabel("pre-trend informativeness  (phi)"); a.set_ylabel("residual violation")

a = ax[1, 0]
a.axhline(95, color="green", lw=.9, ls="--", label="95% nominal")
a.plot(info_grid, cov_naive, "o-", color="#e67e22", label="naive CI")
a.plot(info_grid, cov_split, "s-", color="#16a085", label="sample-split CI")
a.set_title("Causal coverage of the selected LATT")
a.set_xlabel("pre-trend informativeness  (phi)"); a.set_ylabel("coverage (%)"); a.set_ylim(0, 100); a.legend(fontsize=8)

a = ax[1, 1]
a.plot(info_grid, conf_included, "^-", color="#555555")
a.set_title("Mechanism: confounded cohorts slipping into selection\n(out of 6 confounded)")
a.set_xlabel("pre-trend informativeness  (phi)"); a.set_ylabel("avg # confounded selected")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("master_axis.png", dpi=130)
print("\nSaved figure -> master_axis.png")
