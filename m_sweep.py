"""
Fine M-sweep: pin down the composition-randomness finding.
Coverage of the selected LATT vs the smoothness bound M, full-data vs split, with the
residual-curvature distribution of the (random) selected set marked. Claim to confirm:
coverage reaches 95% near the WORST-CASE (max/high-percentile) residual curvature, not
the mean -> M must be calibrated to the worst-case selected composition.
"""
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

TARGET_E = 3
l_post = np.zeros(L2.npost); l_post[TARGET_E-1] = 1.0
L2.V_VEC, L2.W_PRE = L2.extrapolation_weights(l_post)
bbar1 = L2.max_bias_SD(L2.V_VEC, 1.0)
npre, npost = L2.npre, L2.npost
z = norm.ppf(0.975); pre_e, post_e = L2.pre_e, L2.post_e

def cohort_delta(C): return np.array([0.5*C*(e**2) for e in np.concatenate([pre_e, post_e])])
def within_cov(sigma, rho, n):
    idx=np.arange(n); R=rho**np.abs(idx[:,None]-idx[None,:]); return (sigma**2)*R

C_marg, C_dirty, sigma, rho, c_sel = 0.10, 0.35, 0.10, 0.5, 0.80
G = 12
curv = np.array([0]*4 + [C_marg]*4 + [C_dirty]*4, float)
tau = np.concatenate([np.zeros(npre), np.ones(npost)])
means = np.array([tau + cohort_delta(c) for c in curv])
Sc = within_cov(sigma, rho, npre+npost); Lc = np.linalg.cholesky(Sc)
rng = np.random.default_rng(5)

M_grid = np.linspace(0.0, 0.09, 31)
n_reps = 10000

# pre-draw selections to also collect residual-curvature distribution
resid_list = []
cov_full = {M:0 for M in M_grid}; cov_split = {M:0 for M in M_grid}
for _ in range(n_reps):
    B = means + (Lc @ rng.standard_normal((npre+npost, G))).T
    sel = np.max(np.abs(B[:, :npre]), axis=1) <= c_sel
    if sel.sum()==0: sel = np.ones(G, bool)
    resid_list.append(curv[sel].mean())
    agg = B[sel].mean(0); sv = np.sqrt(L2.V_VEC @ (Sc/sel.sum()) @ L2.V_VEC); c = float(L2.V_VEC@agg)
    B1 = means + (Lc @ rng.standard_normal((npre+npost, G))).T*np.sqrt(2)
    B2 = means + (Lc @ rng.standard_normal((npre+npost, G))).T*np.sqrt(2)
    s2 = np.max(np.abs(B1[:, :npre]), axis=1) <= c_sel
    if s2.sum()==0: s2 = np.ones(G, bool)
    agg2 = B2[s2].mean(0); sv2 = np.sqrt(L2.V_VEC @ (2*Sc/s2.sum()) @ L2.V_VEC); c2 = float(L2.V_VEC@agg2)
    for M in M_grid:
        h = L2.cv((M*bbar1)/sv)*sv;  cov_full[M]  += (c-h <= 1.0 <= c+h)
        h2 = L2.cv((M*bbar1)/sv2)*sv2; cov_split[M] += (c2-h2 <= 1.0 <= c2+h2)

resid = np.array(resid_list)
cf = np.array([100*cov_full[M]/n_reps for M in M_grid])
cs = np.array([100*cov_split[M]/n_reps for M in M_grid])
r_mean, r_p95, r_max = resid.mean(), np.percentile(resid, 95), resid.max()

def crossM(cov):
    ok = np.where(cov >= 95)[0]
    return M_grid[ok[0]] if len(ok) else np.nan
print(f"residual curvature: mean={r_mean:.3f}  p95={r_p95:.3f}  max={r_max:.3f}")
print(f"M for 95% coverage: full={crossM(cf):.3f}  split={crossM(cs):.3f}")
print(f"max |full-split| coverage gap across M = {np.max(np.abs(cf-cs)):.1f} pp")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("Fine M-sweep: a fixed smoothness bound must be calibrated to the WORST-CASE selected set",
             fontsize=11)
a = ax[0]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(M_grid, cf, "o-", color="#2471a3", label="full-data")
a.plot(M_grid, cs, "s-", color="#16a085", label="split")
a.axvline(r_mean, color="#e67e22", ls=":", lw=1.2, label=f"mean resid curv={r_mean:.3f}")
a.axvline(r_p95, color="#8e44ad", ls=":", lw=1.2, label=f"p95 resid curv={r_p95:.3f}")
a.axvline(r_max, color="#c0392b", ls=":", lw=1.2, label=f"max resid curv={r_max:.3f}")
a.set_title("Coverage vs M (full-data ~ split throughout)")
a.set_xlabel("smoothness bound M"); a.set_ylabel("coverage (%)"); a.set_ylim(40, 102)
a.legend(fontsize=7)

a = ax[1]
a.hist(resid, bins=30, color="#2471a3", alpha=.7)
a.axvline(r_mean, color="#e67e22", ls=":", lw=1.5, label=f"mean={r_mean:.3f}")
a.axvline(r_max, color="#c0392b", ls=":", lw=1.5, label=f"max={r_max:.3f}")
a.set_title("Residual curvature of the RANDOM selected set\n(mean-calibrated M undercovers the high tail)")
a.set_xlabel("residual curvature of selected aggregate"); a.set_ylabel("frequency"); a.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig("m_sweep.png", dpi=130)
print("Saved -> m_sweep.png")
