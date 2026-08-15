"""
Prototype: a LEVEL-bound honest band vs the SD(M) curvature band
----------------------------------------------------------------
Your definition: PT credible = flat pre-difference; the honest object bounds how
far the post-treatment difference can wander from flat. That is a LEVEL restriction

    Delta_Level(M) = { delta : |delta_post(e)| <= M for all post e },

fed to the same Rambachan-Roth machine. The estimator is the raw post-average
(no pre-extrapolation, because we are not assuming smoothness), so

    max bias over Delta_Level(M) = M * sum|l_post| = M          (l_post = ones/npost)
    FLCI = post-avg  +/-  cv(M / sigma) * sigma,   sigma = SE(post-avg).

We compare, on the same data, against the SD(M) band (curvature, linear
extrapolation) already in the paper. Key points to read off:
  - the level band's M and breakdown M* are in OUTCOME units (interpretable);
  - the SD band's M is in curvature units (abstract);
  - each band is honest against its own class; they respond differently to
    different confound shapes.
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

npre, npost = L2.npre, L2.npost
pre_e, post_e = L2.pre_e, L2.post_e
l_post = np.ones(npost) / npost
z = norm.ppf(0.975)


def within_cov(sig, rho, n):
    idx = np.arange(n); R = rho ** np.abs(idx[:, None] - idx[None, :]); return (sig ** 2) * R


def min_var_sd_v(Sigma):
    """Minimum-variance affine SD(M) estimator: v=(w_pre,l_post) minimizing v'Sig v subject to
    cancelling the through-reference linear trend (finite SD bias). This is the estimator the paper
    actually uses, tighter than naive OLS extrapolation."""
    Spp = Sigma[:npre, :npre]; Spq = Sigma[:npre, npre:]
    Sppinv = np.linalg.inv(Spp)
    a = pre_e.astype(float); b = -(l_post @ post_e)      # constraint a'w = b (slope-orthogonality)
    w0 = -Sppinv @ (Spq @ l_post)
    lam = (b - a @ w0) / (a @ Sppinv @ a)
    w = w0 + Sppinv @ a * lam
    return np.concatenate([w, l_post])


# ---------- level-bound FLCI ----------
def flci_level(bhat, Sigma, M):
    center = l_post @ bhat[npre:]
    sigma = np.sqrt(l_post @ Sigma[npre:, npre:] @ l_post)
    bias = M * np.sum(np.abs(l_post))          # = M
    half = L2.cv(bias / sigma) * sigma
    return center, half


# ---------- SD-bound FLCI (paper's current band, min-variance affine estimator) ----------
V = None; bbar1 = None                          # set once Sigma is known (below)
def flci_sd(bhat, Sigma, M):
    center = V @ bhat
    sigma = np.sqrt(V @ Sigma @ V)
    bias = M * bbar1                           # max_bias_SD is linear in M; avoid re-running the LP
    half = L2.cv(bias / sigma) * sigma
    return center, half


# ===== setup: true effect 1, flat pre, known Sigma =====
sigma_c, rho = 0.14, 0.5
Sigma = within_cov(sigma_c, rho, npre + npost)
V = min_var_sd_v(Sigma)                          # min-variance affine SD estimator (the paper's)
bbar1 = L2.max_bias_SD(V, 1.0)
theta = 1.0
beta_mean = np.concatenate([np.zeros(npre), np.full(npost, theta)])   # flat pre, effect 1 post

se_level = np.sqrt(l_post @ Sigma[npre:, npre:] @ l_post)
se_sd = np.sqrt(V @ Sigma @ V)
print("=== estimator SEs (flat pre) ===")
print(f"  level band  (raw post-avg):       SE = {se_level:.4f}")
print(f"  SD band     (linear extrapolation): SE = {se_sd:.4f}   (extrapolation inflates variance)")
print(f"  SD curvature amplification bbar1 = {bbar1:.2f}")

# ---- breakdown M* on the expected (significant) data ----
def breakdown(flci_fn, Mmax, bhat, Sigma):
    Ms = np.linspace(0, Mmax, 4001)
    for M in Ms:
        c, h = flci_fn(bhat, Sigma, M)
        if c - h <= 0.0:            # band first admits the null
            return M
    return np.nan

Mstar_level = breakdown(flci_level, 3.0, beta_mean, Sigma)
Mstar_sd = breakdown(flci_sd, 0.6, beta_mean, Sigma)
print("\n=== breakdown values on the significant effect (=1) ===")
print(f"  level band M* = {Mstar_level:.3f}  (outcome units: a post-violation this large overturns it)")
print(f"  SD band    M* = {Mstar_sd:.3f}  (curvature units)")

# ---- validity: does each band cover iff true violation is within its class? ----
def coverage(flci_fn, delta_true, M, n_reps=6000, seed=1):
    rng = np.random.default_rng(seed); Lc = np.linalg.cholesky(Sigma)
    mean = np.concatenate([np.zeros(npre), np.full(npost, theta)]) + delta_true
    cov = 0
    for _ in range(n_reps):
        b = mean + Lc @ rng.standard_normal(npre + npost)
        c, h = flci_fn(b, Sigma, M)
        cov += (c - h <= theta <= c + h)
    return 100 * cov / n_reps

# constant post shift V0 (flat pre): level violation = V0 ; also has curvature V0 at the kink
print("\n=== validity: constant post shift (flat pre), true post-violation V0=0.30 ===")
d_const = np.concatenate([np.zeros(npre), np.full(npost, 0.30)])
for M in [0.20, 0.30, 0.45]:
    print(f"  M={M:.2f}: level cover={coverage(flci_level, d_const, M):5.1f}%   "
          f"SD cover={coverage(flci_sd, d_const, M):5.1f}%")

# post drift v0*e (flat pre): level violation = v0*4 at e=4 ; curvature (kink) = v0
v0 = 0.08
d_drift = np.concatenate([np.zeros(npre), v0 * post_e.astype(float)])
print(f"\n=== validity: post drift v0*e, v0={v0} (level viol at e=4 = {v0*4:.2f}, kink curvature = {v0}) ===")
for M in [0.10, 0.20, 0.35]:
    print(f"  M={M:.2f}: level cover={coverage(flci_level, d_drift, M):5.1f}%   "
          f"SD cover={coverage(flci_sd, d_drift, M):5.1f}%")

# ---------- figure: the two bands as M relaxes ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("Level-bound band vs SD(M) curvature band on the same significant effect (=1)", fontsize=11)

a = ax[0]
Ms = np.linspace(0, 1.2, 200)
cen = np.array([flci_level(beta_mean, Sigma, M) for M in Ms])
a.fill_between(Ms, cen[:, 0] - cen[:, 1], cen[:, 0] + cen[:, 1], color="#2471a3", alpha=.25)
a.plot(Ms, cen[:, 0] - cen[:, 1], color="#2471a3"); a.plot(Ms, cen[:, 0] + cen[:, 1], color="#2471a3")
a.axhline(0, color="gray", lw=.8, ls=":"); a.axvline(Mstar_level, color="#c0392b", ls="--", lw=1)
a.set_title(f"Level band  (M in outcome units)\nbreakdown M* = {Mstar_level:.2f}")
a.set_xlabel("M = allowed post-treatment PT violation"); a.set_ylabel("LATT confidence band")

a = ax[1]
Ms2 = np.linspace(0, 0.12, 200)
cen2 = np.array([flci_sd(beta_mean, Sigma, M) for M in Ms2])
a.fill_between(Ms2, cen2[:, 0] - cen2[:, 1], cen2[:, 0] + cen2[:, 1], color="#16a085", alpha=.25)
a.plot(Ms2, cen2[:, 0] - cen2[:, 1], color="#16a085"); a.plot(Ms2, cen2[:, 0] + cen2[:, 1], color="#16a085")
a.axhline(0, color="gray", lw=.8, ls=":"); a.axvline(Mstar_sd, color="#c0392b", ls="--", lw=1)
a.set_title(f"SD(M) curvature band\nbreakdown M* = {Mstar_sd:.3f} (curvature units)")
a.set_xlabel("M = allowed curvature"); a.set_ylabel("LATT confidence band")

plt.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig("level_vs_sd.png", dpi=130)
print("\nSaved -> level_vs_sd.png")
