"""
Proposition 11 (optimal credible weighting) -- simulation
---------------------------------------------------------
Reduced-form normal model, 4+4 event study, homogeneous effects (theta_g = 1),
so ATT = LATT target = 1 and mean squared error for the common target is a fair
comparison between procedures.

We compare three ways of weighting K cohorts whose credibility varies
continuously:
    ATT          equal weights over ALL cohorts
    Hard screen  equal weights over {g : m_g <= c}, swept over c
    Optimal      lambda* minimizing worst-case MSE  R(lam) = (m.lam)^2 + sum lam^2 s_g^2
                 over the simplex (Proposition 11), in two versions:
                 - oracle:   credibility m_g known (= true violation V_g)
                 - feasible: m_hat_g = max_e |beta_hat_pre(e)| estimated from pre-trends

Under homogeneous effects the estimand is common (=1), so for fixed weights
    MSE(lam) = (sum_g lam_g V_g)^2 + sum_g lam_g^2 s_g^2,
which is exact; the oracle-optimal lambda* therefore minimizes MSE by construction.
The feasible version pays for estimating m_g from noisy pre-trends and is the
practically relevant comparison.

DGP is deterministic (a credibility gradient with a few moderate-violation but
PRECISE cohorts, to exhibit the precision channel). A random-DGP loop at the end
reports the average gain for generality.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

# ---- event-study geometry (matches the rest of the paper: 4 pre, 4 post) ----
K_pre, L_post = 4, 4
l_post = np.ones(L_post) / L_post
rho = 0.5


def post_var(sigma_g):
    """Variance of the post-aggregate l_post' beta_post under within-cohort AR(1)."""
    idx = np.arange(L_post)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    return float(l_post @ ((sigma_g ** 2) * R) @ l_post)


def pre_cov(sigma_g):
    idx = np.arange(K_pre)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    return (sigma_g ** 2) * R


def mse(lam, V, s2):
    """Exact MSE of sum_g lam_g beta_g,post for the common target 1 (theta_g=1)."""
    bias = lam @ V
    return bias ** 2 + np.sum(lam ** 2 * s2)


def opt_weights(m, s2):
    """Minimize R(lam) = (m.lam)^2 + sum lam^2 s2 over the simplex (Prop 11)."""
    G = len(m)
    Q = np.outer(m, m) + np.diag(s2)
    obj = lambda lam: lam @ Q @ lam
    jac = lambda lam: 2 * Q @ lam
    cons = ({"type": "eq", "fun": lambda lam: lam.sum() - 1.0,
             "jac": lambda lam: np.ones(G)},)
    bnds = [(0.0, 1.0)] * G
    x0 = (1.0 / s2) / np.sum(1.0 / s2)          # inverse-variance start
    res = minimize(obj, x0, jac=jac, bounds=bnds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    lam = np.clip(res.x, 0, None)
    return lam / lam.sum()


def hard_weights(m, c):
    sel = m <= c
    if not sel.any():
        sel = (m == m.min())
    w = sel.astype(float)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Deterministic illustrative DGP. Cohorts come in PAIRS at each violation
# level: within a pair one cohort is precise (large n) and one is noisy
# (small n). A hard screen keeps or drops a pair as a whole -- it cannot tell
# the precise member from the noisy one, since both share the credibility m_g.
# The optimal weighting can, and it also soft-thresholds the gradient rather
# than cutting it. Pre-trends are informative (small SE relative to V).
# ---------------------------------------------------------------------------
levels = np.array([0.00, 0.08, 0.16, 0.24, 0.32, 0.40])
V = np.repeat(levels, 2)                          # 12 cohorts, paired by violation
G = len(V)
sigma_unit = 0.12
# within each pair: member A precise (n=6400), member B noisy (n=800)
n_g = np.tile([6400.0, 800.0], len(levels))
sigma_g = sigma_unit / np.sqrt(n_g / 400.0)      # sigma scales as 1/sqrt(size)
s2 = np.array([post_var(sg) for sg in sigma_g])  # post-aggregate variances

# ---- procedures (oracle: credibility m = true violation V) ----
w_att = np.ones(G) / G
lam_opt = opt_weights(V, s2)
mse_att = mse(w_att, V, s2)
mse_opt = mse(lam_opt, V, s2)

# hard screen swept over threshold c
c_grid = np.linspace(0.0, 0.70, 60)
mse_hard = np.array([mse(hard_weights(V, c), V, s2) for c in c_grid])
c_best = c_grid[np.argmin(mse_hard)]
mse_hard_best = mse_hard.min()
lam_hard_best = hard_weights(V, c_best)

# ---- feasible optimal: estimate m_hat from noisy pre-trends, informativeness phi ----
phi = 1.5            # pre-trend foreshadows the violation (informative regime)
n_reps = 4000
pre_true = np.array([np.full(K_pre, phi * V[g]) for g in range(G)])   # E[beta_pre]
Lpre = [np.linalg.cholesky(pre_cov(sg)) for sg in sigma_g]

# feasible procedures pay the cost of estimating credibility from pre-trends.
# For a fair comparison, the feasible hard screen also uses m_hat, swept over c.
mse_feas_opt = np.empty(n_reps)
mse_feas_hard_c = np.zeros(len(c_grid))          # feasible hard MSE per threshold c
lam_feas_acc = np.zeros(G)
for r in range(n_reps):
    beta_pre = np.array([pre_true[g] + Lpre[g] @ rng.standard_normal(K_pre) for g in range(G)])
    m_hat = np.max(np.abs(beta_pre), axis=1)
    lam_f = opt_weights(m_hat, s2)
    mse_feas_opt[r] = mse(lam_f, V, s2)          # TRUE mse of feasible weights
    mse_feas_hard_c += np.array([mse(hard_weights(m_hat, c), V, s2) for c in c_grid])
    lam_feas_acc += lam_f
lam_feas_mean = lam_feas_acc / n_reps
mse_feas_opt_mean = mse_feas_opt.mean()
mse_feas_hard_c /= n_reps
mse_feas_hard_best = mse_feas_hard_c.min()       # best feasible hard screen

print("=== Proposition 11 demonstration (deterministic DGP) ===")
print(f"{'procedure':<32}{'MSE':>10}{'RMSE':>10}{'vs ATT':>10}")
for name, m in [("ATT (all cohorts)", mse_att),
                ("hard screen, oracle best c", mse_hard_best),
                ("hard screen, feasible best c", mse_feas_hard_best),
                ("optimal, feasible (m_hat)", mse_feas_opt_mean),
                ("optimal, oracle (m=V)", mse_opt)]:
    print(f"{name:<32}{m:10.5f}{np.sqrt(m):10.5f}{100*(m/mse_att-1):+9.1f}%")
print(f"\nbest oracle hard-screen threshold c* = {c_best:.3f}")
print(f"optimal (oracle)   vs best ORACLE   hard screen: {100*(1-mse_opt/mse_hard_best):+.1f}%")
print(f"optimal (feasible) vs best FEASIBLE hard screen: {100*(1-mse_feas_opt_mean/mse_feas_hard_best):+.1f}%")

# ---------------------------------------------------------------------------
# Random-DGP robustness: average gain of oracle-optimal over best hard screen
# ---------------------------------------------------------------------------
n_dgp = 500
red_opt, red_feas = [], []
for _ in range(n_dgp):
    Vr = np.sort(rng.uniform(0, 0.6, G))
    Vr[:rng.integers(1, 4)] = 0.0                        # a few clean cohorts
    nr = rng.uniform(300, 1800, G)
    sgr = sigma_unit / np.sqrt(nr / 400.0)
    s2r = np.array([post_var(sg) for sg in sgr])
    mo = mse(opt_weights(Vr, s2r), Vr, s2r)
    mh = min(mse(hard_weights(Vr, c), Vr, s2r) for c in c_grid)
    red_opt.append(1 - mo / mh)
print(f"\nRandom DGPs (n={n_dgp}): mean MSE reduction of oracle-optimal over "
      f"best hard screen = {100*np.mean(red_opt):.1f}% "
      f"(median {100*np.median(red_opt):.1f}%)")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle("Proposition 11: the soft optimal weighting improves on the hard flatness screen\n"
             "(homogeneous effects, so every procedure targets the same value; MSE for that target)",
             fontsize=11)

a = ax[0]
a.plot(c_grid, mse_hard, "-", color="#7d3c98", lw=2, label="hard screen (oracle), vs threshold $c$")
a.axhline(mse_att, color="#c0392b", ls="--", lw=1.5, label="ATT (all cohorts)")
a.axhline(mse_opt, color="#2471a3", ls="-", lw=1.8, label="optimal, oracle $m_g=V_g$")
a.axhline(mse_feas_opt_mean, color="#16a085", ls=":", lw=1.8, label=r"optimal, feasible $\hat m_g$")
a.plot([c_best], [mse_hard_best], "o", color="#7d3c98", ms=7)
a.annotate("best $c$", (c_best, mse_hard_best),
           textcoords="offset points", xytext=(6, 8), fontsize=8, color="#7d3c98")
a.set_xlabel("hard-screen threshold $c$"); a.set_ylabel("MSE for the common target")
a.set_title("MSE: optimal weighting beats the hard screen at every $c$")
a.legend(fontsize=8, loc="upper right")

a = ax[1]
# grouped by violation level: precise member (even idx) vs noisy member (odd idx)
xl = np.arange(len(levels))
opt_prec = lam_opt[0::2]     # precise member of each pair
opt_nois = lam_opt[1::2]     # noisy member
hard_prec = lam_hard_best[0::2]
hard_nois = lam_hard_best[1::2]
a.bar(xl - 0.22, opt_prec, width=0.2, color="#2471a3", label="optimal $\\lambda^*$: precise member")
a.bar(xl - 0.02, opt_nois, width=0.2, color="#7fb3d5", label="optimal $\\lambda^*$: noisy member")
a.bar(xl + 0.20, (hard_prec + hard_nois) / 2, width=0.2, color="#7d3c98", alpha=0.7,
      label=f"hard screen at $c^*$={c_best:.2f} (per member)")
a.set_xlabel("violation level $V_g$ (credibility; lower = more credible)")
a.set_ylabel("weight $\\lambda_g$")
a.set_title("The optimal weighting favors the precise member\nand soft-thresholds the gradient")
a.set_xticks(xl); a.set_xticklabels([f"{v:.2f}" for v in levels], fontsize=8)
a.legend(fontsize=7.5, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("prop11.png", dpi=130)
print("\nSaved figure -> prop11.png")
