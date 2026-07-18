"""
Full Layer 2 headline: SD(M) FLCI restores honest coverage of the causal target
where the point estimate collapses. Target = post-period effect tau_3.

The differential trend (residual confound after imperfect selection) is a smooth
curve of curvature C. Point estimates (naive, and linear-extrapolation) are unbiased
only when C=0 and collapse as C grows; the FLCI stays honest exactly when M >= C,
paying with width. This is the smoothness class -- it leverages STRUCTURE, not
pre-trend magnitude, so it does not inherit the relative-magnitudes informativeness trap.
"""
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

# target a later post-period tau_3 (curvature bias substantial -> point estimates collapse)
TARGET_E = 3
l_post = np.zeros(L2.npost); l_post[TARGET_E - 1] = 1.0
L2.V_VEC, L2.W_PRE = L2.extrapolation_weights(l_post)

npre, npost = L2.npre, L2.npost
sigma_coef = 0.10
Sigma = (sigma_coef ** 2) * np.eye(npre + npost)
theta_true = 1.0
tau = np.concatenate([np.zeros(npre), np.full(npost, theta_true)])
z = norm.ppf(0.975)
rng = np.random.default_rng(303)
n_reps = 3000

# b_bar scales linearly in M: b_bar(M) = M * b_bar(1). Cache once (LP is data-independent).
bbar1 = L2.max_bias_SD(L2.V_VEC, 1.0)
sigma_v = np.sqrt(float(L2.V_VEC @ Sigma @ L2.V_VEC))
print(f"target tau_3: max-bias per unit M = {bbar1:.3f}, sigma_v = {sigma_v:.3f}")

def flci_fast(bhat, M):
    ctr = float(L2.V_VEC @ bhat)
    half = L2.cv((M * bbar1) / sigma_v) * sigma_v
    return ctr, half

C_grid = np.linspace(0.0, 0.25, 14)
M_lo, M_hi = 0.10, 0.20

cov_naive, cov_lin, cov_flo, cov_fhi = [], [], [], []
half_flo, half_fhi = [], []
for C in C_grid:
    delta = L2.delta_quadratic(C)
    beta_mean = tau + delta
    cn = cl = cflo = cfhi = 0; hlo = hhi = 0.0
    for _ in range(n_reps):
        bhat = beta_mean + sigma_coef * rng.standard_normal(npre + npost)
        # naive point: target post-period coefficient, no Delta
        mn = bhat[npre + TARGET_E - 1]; se = sigma_coef
        cn += (mn - z*se <= theta_true <= mn + z*se)
        # linear-extrapolation point, sampling-only CI
        le = float(L2.V_VEC @ bhat); sle = np.sqrt(L2.V_VEC @ Sigma @ L2.V_VEC)
        cl += (le - z*sle <= theta_true <= le + z*sle)
        # FLCI at two M
        ctr, h = flci_fast(bhat, M_lo); cflo += (ctr-h <= theta_true <= ctr+h); hlo += h
        ctr, h = flci_fast(bhat, M_hi); cfhi += (ctr-h <= theta_true <= ctr+h); hhi += h
    cov_naive.append(100*cn/n_reps); cov_lin.append(100*cl/n_reps)
    cov_flo.append(100*cflo/n_reps); cov_fhi.append(100*cfhi/n_reps)
    half_flo.append(hlo/n_reps); half_fhi.append(hhi/n_reps)

cov_naive, cov_lin = np.array(cov_naive), np.array(cov_lin)
cov_flo, cov_fhi = np.array(cov_flo), np.array(cov_fhi)

# sensitivity/breakdown: one representative dataset at C=0.10, sweep M
C0 = 0.10
delta0 = L2.delta_quadratic(C0); beta0 = tau + delta0
# average many draws to get the expected interval vs M
M_sweep = np.linspace(0.0, 0.30, 40)
ctrs, halves = [], []
for M in M_sweep:
    cc = []; hh = []
    for _ in range(1500):
        bhat = beta0 + sigma_coef * rng.standard_normal(npre + npost)
        ctr, h = flci_fast(bhat, M)
        cc.append(ctr); hh.append(h)
    ctrs.append(np.mean(cc)); halves.append(np.mean(hh))
ctrs, halves = np.array(ctrs), np.array(halves)
lo_band, hi_band = ctrs - halves, ctrs + halves
covers0 = (lo_band <= theta_true) & (theta_true <= hi_band)
Mstar = M_sweep[np.argmax(covers0)] if covers0.any() else np.nan
excl0 = hi_band < 0.0   # would we (wrongly) still exclude a null? here effect>0

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Full Layer 2 (validated): SD(M) FLCI restores honest coverage where point estimates collapse",
             fontsize=12)

a = ax[0]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(C_grid, cov_naive, "o-", color="#c0392b", label="naive point")
a.plot(C_grid, cov_lin, "d-", color="#e67e22", label="linear-extrap point")
a.plot(C_grid, cov_flo, "s-", color="#2471a3", label=f"FLCI, M={M_lo}")
a.plot(C_grid, cov_fhi, "^-", color="#16a085", label=f"FLCI, M={M_hi}")
a.axvline(M_lo, color="#2471a3", ls=":", lw=.8); a.axvline(M_hi, color="#16a085", ls=":", lw=.8)
a.set_title("Coverage of tau_3 vs true curvature C\n(FLCI honest while C <= M)")
a.set_xlabel("true confound curvature C"); a.set_ylabel("coverage (%)"); a.set_ylim(0, 103)
a.legend(fontsize=7)

a = ax[1]
a.plot(C_grid, half_flo, "s-", color="#2471a3", label=f"FLCI half-width, M={M_lo}")
a.plot(C_grid, half_fhi, "^-", color="#16a085", label=f"FLCI half-width, M={M_hi}")
a.set_title("FLCI width: pay for honesty with wider intervals\n(width set by assumed M, ~flat in C)")
a.set_xlabel("true confound curvature C"); a.set_ylabel("half-width"); a.legend(fontsize=8)

a = ax[2]
a.fill_between(M_sweep, lo_band, hi_band, alpha=.25, color="#2471a3", label="FLCI band")
a.axhline(theta_true, color="black", lw=1.2, label="true tau_3 = 1")
a.axhline(0, color="gray", ls=":", lw=.8)
if np.isfinite(Mstar):
    a.axvline(Mstar, color="#c0392b", ls="--", lw=1, label=f"covers truth for M>={Mstar:.2f}")
a.set_title(f"Sensitivity / breakdown (data at C={C0})")
a.set_xlabel("assumed smoothness bound M"); a.set_ylabel("tau_3"); a.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("layer2_full.png", dpi=130)
print("Saved -> layer2_full.png")
print(f"breakdown M* (data at C={C0}) = {Mstar:.3f}")
