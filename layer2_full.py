"""
Full Layer 2: honest set-valued inference on the selected LATT via HonestDiD-style
FLCI for the smoothness class SD(M).

Estimator: linear extrapolation of the pre-trend (the interpretable RR/AK affine
estimator). Fit a line to the pre-period coefficients, extrapolate to the post
target, subtract. Valid if the differential trend is exactly linear (SD(0)); SD(M)
relaxes to "approximately linear" (|second differences| <= M).

FLCI = theta_hat  +/-  cv_{1-alpha}( Mbar_bias / sigma ) * sigma,
  where theta_hat = l'beta_post + w_pre' beta_pre  (w_pre = -extrapolation weights),
        Mbar_bias = max_{delta in SD(M)} |v'delta|,  v = (w_pre, l),
        sigma^2   = v' Sigma v,
        cv_{1-alpha}(t) = (1-alpha) quantile of |N(t,1)| (folded normal).

Stage 1 here: validate cv(.) and the max-bias LP, and confirm the estimator
differences out linear trends (so bias is finite under SD).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, linprog

# ---------- event-time grid ----------
K, L = 4, 4                     # pre periods e=-K..-1 ; post e=1..L ; reference e=0 (delta_0=0)
pre_e = np.arange(-K, 0)        # -4..-1
post_e = np.arange(1, L + 1)    # 1..4
all_e = np.concatenate([pre_e, [0], post_e])     # includes reference 0
npre, npost = K, L


# ---------- Stage 1a: folded-normal critical value ----------
def cv(t, alpha=0.05):
    """(1-alpha) quantile of |N(t,1)|: solve P(-q<=N(t,1)<=q)=1-alpha for q."""
    t = abs(t)
    f = lambda q: (norm.cdf(q - t) - norm.cdf(-q - t)) - (1 - alpha)
    lo, hi = 0.0, t + 10.0
    return brentq(f, lo, hi)


# ---------- Stage 1b: linear-extrapolation weights ----------
def extrapolation_weights(l_post):
    """
    theta_hat = l'beta_post - l'(linear extrapolation of beta_pre to post times).
    Fit line (intercept,slope) by OLS to pre points (e in pre_e); extrapolate to post_e.
    Returns v = (w_pre, l_post) as a length (npre+npost) vector so that
    E[theta_hat] - theta = v' delta.
    """
    # OLS design on pre periods: X = [1, e]
    X = np.column_stack([np.ones(npre), pre_e])
    XtX_inv = np.linalg.inv(X.T @ X)
    # predicted line at post times: Xpost @ (XtX_inv X') beta_pre
    Xpost = np.column_stack([np.ones(npost), post_e])
    H = Xpost @ XtX_inv @ X.T            # (npost, npre): maps beta_pre -> extrapolated post
    # theta_hat = l' beta_post - l' H beta_pre  => w_pre = -H' l
    w_pre = -(H.T @ l_post)
    v = np.concatenate([w_pre, l_post])
    return v, w_pre


# ---------- Stage 1c: max bias over SD(M) ----------
def max_bias_SD(v, M):
    """
    max_{delta in SD(M)} |v'delta|, delta indexed by all_e with delta_0 = 0.
    SD(M): |delta_{e+1} - 2 delta_e + delta_{e-1}| <= M for interior e.
    delta has length len(all_e); the reference (e=0) entry is fixed to 0.
    v aligns with (pre_e, post_e) (excludes e=0). Solve LP; return +inf if unbounded.
    """
    idx = {e: i for i, e in enumerate(all_e)}
    n = len(all_e)
    # variables: delta over all_e, with delta_0 = 0 enforced as equality
    # objective vector c aligns v (pre then post) onto all_e positions
    c = np.zeros(n)
    for j, e in enumerate(pre_e):
        c[idx[e]] = v[j]
    for j, e in enumerate(post_e):
        c[idx[e]] = v[npre + j]
    # second-difference constraints for interior e (need e-1,e,e+1 in grid)
    A = []
    for e in all_e:
        if (e - 1) in idx and (e + 1) in idx:
            row = np.zeros(n)
            row[idx[e - 1]] += 1; row[idx[e]] += -2; row[idx[e + 1]] += 1
            A.append(row)
    A = np.array(A)
    d = np.full(len(A), M)
    # |A delta| <= M  ->  A delta <= M and -A delta <= M
    A_ub = np.vstack([A, -A]); b_ub = np.concatenate([d, d])
    # equality: delta_0 = 0
    A_eq = np.zeros((1, n)); A_eq[0, idx[0]] = 1; b_eq = [0.0]
    bounds = [(None, None)] * n
    # maximize c'delta  == minimize -c'delta
    res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res_max.status == 3 or res_min.status == 3:   # unbounded
        return np.inf
    hi = -res_max.fun      # max c'delta
    lo = res_min.fun       # min c'delta
    return max(abs(hi), abs(lo))


# ---------- Stage 2: the FLCI ----------
l_post = np.ones(npost) / npost                 # target: average post-treatment effect
V_VEC, W_PRE = extrapolation_weights(l_post)


def flci(bhat, Sigma, M, alpha=0.05):
    """
    bhat = (beta_pre[-K..-1], beta_post[1..L]) ~ N(tau+delta, Sigma).
    Returns (center, half_length) of the SD(M) FLCI for theta = l'tau_post.
    """
    theta_hat = float(V_VEC @ bhat)
    bbar = max_bias_SD(V_VEC, M)
    sigma = np.sqrt(float(V_VEC @ Sigma @ V_VEC))
    if not np.isfinite(bbar):
        return theta_hat, np.inf
    half = cv(bbar / sigma, alpha) * sigma
    return theta_hat, half


def delta_quadratic(C):
    """confound path delta_e = (C/2) e^2 -> max|second difference| = C exactly, delta_0=0."""
    return np.array([0.5 * C * (e ** 2) for e in np.concatenate([pre_e, post_e])])


rng = np.random.default_rng(11)

if __name__ == "__main__":
    print("Stage 1a: cv(t)")
    for t in [0.0, 1.0, 2.0]:
        print(f"   cv_.95({t}) = {cv(t):.4f}")
    print("Stage 1b/c: v'delta(linear)=%.1e ; maxbias SD(1)=%.3f" %
          (V_VEC @ np.concatenate([pre_e, post_e]), max_bias_SD(V_VEC, 1.0)))

    # ---- Stage 2 validation: does FLCI(M) cover theta=1 iff true curvature C <= M? ----
    sigma_coef = 0.5
    Sigma = (sigma_coef ** 2) * np.eye(npre + npost)
    theta_true = 1.0
    tau = np.concatenate([np.zeros(npre), np.full(npost, theta_true)])   # tau_pre=0, tau_post=1
    n_reps = 4000
    z = norm.ppf(0.975)

    print("\nStage 2 validation: FLCI(M) coverage of theta=1  (should be >=95% iff C<=M)")
    print(f"  {'C(true)':>8} {'M(assumed)':>10} | {'FLCI cov':>9} {'FLCI half':>10} | "
          f"{'naive cov':>9} {'linextrap cov':>13}")
    for C in [0.0, 0.5, 1.0]:
        delta = delta_quadratic(C)
        beta_mean = tau + delta
        for M in [0.5, 1.0]:
            cov_flci = cov_naive = cov_lin = 0
            half_acc = 0.0
            for _ in range(n_reps):
                bhat = beta_mean + np.linalg.cholesky(Sigma) @ rng.standard_normal(npre + npost)
                ctr, half = flci(bhat, Sigma, M)
                cov_flci += (ctr - half <= theta_true <= ctr + half); half_acc += half
                # naive: no extrapolation, no Delta -> theta_hat = mean(beta_post), se only
                mn = bhat[npre:].mean(); se = sigma_coef / np.sqrt(npost)
                cov_naive += (mn - z*se <= theta_true <= mn + z*se)
                # linear-extrapolation point estimate, sampling-only CI (no Delta allowance)
                lin_est = float(V_VEC @ bhat); se_lin = np.sqrt(V_VEC @ Sigma @ V_VEC)
                cov_lin += (lin_est - z*se_lin <= theta_true <= lin_est + z*se_lin)
            print(f"  {C:8.2f} {M:10.2f} | {100*cov_flci/n_reps:8.1f}% {half_acc/n_reps:10.4f} | "
                  f"{100*cov_naive/n_reps:8.1f}% {100*cov_lin/n_reps:12.1f}%")
