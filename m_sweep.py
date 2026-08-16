"""
Fine M-sweep (level band): calibration to the random selected set.
Coverage of the selected LATT vs the level bound M, full-data vs split, with the
residual-violation distribution of the (random) selected set marked. Claim to confirm:
coverage reaches 95% near the WORST-CASE (high-percentile) residual violation, not the
mean -> M must be calibrated to the worst-case selected composition.

Level/flatness world: confounded cohorts carry a post-treatment level violation; the
FLATNESS screen keeps near-flat cohorts; the level band Delta_Level(M)={|delta_post|<=M}
is applied to the selected aggregate (estimator = raw post-average). M is in outcome units.
"""
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

npre, npost = L2.npre, L2.npost
l_post = np.ones(npost) / npost
z = norm.ppf(0.975)

V_marg, V_dirty, sigma, rho, c_sel = 0.10, 0.35, 0.10, 0.5, 0.22
G = 12
Vvec = np.array([0]*4 + [V_marg]*4 + [V_dirty]*4, float)     # 4 clean, 4 marginal, 4 dirty
tau = np.concatenate([np.zeros(npre), np.ones(npost)])


def confound(Vv, phi):
    return np.concatenate([np.full(npre, phi * Vv), np.full(npost, Vv)])


def within_cov(sig, rr, n):
    idx = np.arange(n); R = rr ** np.abs(idx[:, None] - idx[None, :]); return (sig ** 2) * R


means = np.array([tau + confound(v, 1.0) for v in Vvec])       # phi=1 (violation visible pre and post)
Sc = within_cov(sigma, rho, npre + npost); Lc = np.linalg.cholesky(Sc)
sigma_post1 = np.sqrt(l_post @ Sc[npre:, npre:] @ l_post)
rng = np.random.default_rng(5)

M_grid = np.linspace(0.0, 0.13, 31)
n_reps = 10000

resid_list = []
cov_full = {M: 0 for M in M_grid}; cov_split = {M: 0 for M in M_grid}
for _ in range(n_reps):
    B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
    stat = np.max(np.abs(B[:, :npre]), axis=1)                 # FLATNESS screen
    sel = stat <= c_sel
    if sel.sum() == 0: sel = (stat == stat.min())
    resid_list.append(Vvec[sel].mean())                        # residual post-violation (outcome units)
    center = (B[sel][:, npre:] @ l_post).mean(); sv = sigma_post1 / np.sqrt(sel.sum())
    B1 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
    B2 = means + (Lc @ rng.standard_normal((npre + npost, G))).T * np.sqrt(2)
    stat1 = np.max(np.abs(B1[:, :npre]), axis=1); s2 = stat1 <= c_sel
    if s2.sum() == 0: s2 = (stat1 == stat1.min())
    center2 = (B2[s2][:, npre:] @ l_post).mean(); sv2 = (sigma_post1 * np.sqrt(2)) / np.sqrt(s2.sum())
    for M in M_grid:
        _, h = L2.flci_level(center, sv, M); cov_full[M] += (center - h <= 1.0 <= center + h)
        _, h2 = L2.flci_level(center2, sv2, M); cov_split[M] += (center2 - h2 <= 1.0 <= center2 + h2)

resid = np.array(resid_list)
cf = np.array([100*cov_full[M]/n_reps for M in M_grid])
cs = np.array([100*cov_split[M]/n_reps for M in M_grid])
r_mean, r_p95, r_max = resid.mean(), np.percentile(resid, 95), resid.max()

def crossM(cov):
    ok = np.where(cov >= 95)[0]; return M_grid[ok[0]] if len(ok) else np.nan
print(f"residual violation: mean={r_mean:.3f}  p95={r_p95:.3f}  max={r_max:.3f}")
print(f"M for 95% coverage: full={crossM(cf):.3f}  split={crossM(cs):.3f}")
print(f"max |full-split| coverage gap across M = {np.max(np.abs(cf-cs)):.1f} pp")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("Fine M-sweep (level band): a fixed bound must be calibrated to the WORST-CASE selected set",
             fontsize=11)
a = ax[0]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(M_grid, cf, "o-", color="#2471a3", label="full-data")
a.plot(M_grid, cs, "s-", color="#16a085", label="split")
a.axvline(r_mean, color="#e67e22", ls=":", lw=1.2, label=f"mean resid viol={r_mean:.3f}")
a.axvline(r_p95, color="#8e44ad", ls=":", lw=1.2, label=f"p95 resid viol={r_p95:.3f}")
a.axvline(r_max, color="#c0392b", ls=":", lw=1.2, label=f"max resid viol={r_max:.3f}")
a.set_title("Coverage vs M (full-data ~ split)")
a.set_xlabel("level bound M (outcome units)"); a.set_ylabel("coverage (%)"); a.set_ylim(40, 102); a.legend(fontsize=7)

a = ax[1]
a.hist(resid, bins=30, color="#2471a3", alpha=.7)
a.axvline(r_mean, color="#e67e22", ls=":", lw=1.5, label=f"mean={r_mean:.3f}")
a.axvline(r_max, color="#c0392b", ls=":", lw=1.5, label=f"max={r_max:.3f}")
a.set_title("Residual post-violation of the RANDOM selected set\n(mean-calibrated M undercovers the high tail)")
a.set_xlabel("residual post-violation of selected aggregate (outcome units)"); a.set_ylabel("frequency"); a.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig("m_sweep.png", dpi=130)
print("Saved -> m_sweep.png")
