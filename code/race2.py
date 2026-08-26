"""
Selective-vs-splitting race, done properly.
-------------------------------------------
Isolate the SELECTION problem (Layer 1) from identification (Layer 2):
  - ALL cohorts are clean in post (delta_post = 0, theta = 1), so the causal target
    for ANY selected set is exactly 1 (no identification gap, ever).
  - Cohorts have heterogeneous TRUE pre-trends spread AROUND the threshold c, so
    selection |beta_hat_pre_g| <= c is genuinely STOCHASTIC and asymmetric -> the
    correlation rho between pre and post biases naive post estimates (selection
    distortion). This is exactly the regime where naive undercovers and the
    selective-vs-splitting question is real.

Selective CI = exact polyhedral (Lee, Sun, Sun & Taylor 2016) truncated-normal
interval for the linear contrast eta'beta = mean of beta_post over the selected set,
conditional on the selection event (box on the pre-coefficients, signs of excluded
cohorts conditioned on).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20240719)
z = norm.ppf(0.975)


def _phidiff(lo, hi):
    """Phi(hi)-Phi(lo), numerically stable."""
    if hi <= lo:
        return 0.0
    if hi < 0:
        return norm.cdf(hi) - norm.cdf(lo)
    if lo > 0:
        return norm.sf(lo) - norm.sf(hi)
    return norm.cdf(hi) - norm.cdf(lo)


def selective_interval(bhat, Sigma, eta, constraints, alpha=0.05):
    """
    Exact polyhedral selective CI for theta = eta'mu given bhat ~ N(mu, Sigma),
    conditional on {a'bhat <= b for (a,b) in constraints}.
    Returns (lo, hi, length).
    """
    T = float(eta @ bhat)
    s2 = float(eta @ Sigma @ eta)
    if s2 <= 0:
        return T, T, 0.0
    sig = np.sqrt(s2)
    c_dir = Sigma @ eta / s2                      # direction c = Sigma eta / (eta'Sigma eta)
    Z = bhat - c_dir * T                          # nuisance, fixed under conditioning

    Vlo, Vhi = -np.inf, np.inf
    for a, b in constraints:
        alpha_j = float(a @ c_dir)
        resid = b - float(a @ Z)
        if alpha_j > 1e-12:
            Vhi = min(Vhi, resid / alpha_j)
        elif alpha_j < -1e-12:
            Vlo = max(Vlo, resid / alpha_j)
        # alpha_j ~ 0: constraint independent of theta; must hold, ignore
    # guard: observed T must lie in [Vlo, Vhi]
    Vlo = min(Vlo, T)
    Vhi = max(Vhi, T)

    def pivot(theta0):
        # P(TN <= T) with mean theta0, sd sig, truncated to [Vlo,Vhi]; decreasing in theta0
        a = (Vlo - theta0) / sig
        b = (Vhi - theta0) / sig
        t = (T - theta0) / sig
        denom = _phidiff(a, b)
        if denom <= 1e-300:
            return None
        return _phidiff(a, t) / denom

    def solve(level):
        # find theta0 with pivot(theta0) = level; pivot decreasing in theta0
        lo, hi = T - 40 * sig, T + 40 * sig
        flo, fhi = pivot(lo), pivot(hi)
        if flo is None or fhi is None:
            return np.nan
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = pivot(mid)
            if fm is None:
                # numerical trouble -> widen and bail
                return lo if level > 0.5 else hi
            if fm > level:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    hi_end = solve(alpha / 2)        # upper endpoint (pivot small)
    lo_end = solve(1 - alpha / 2)    # lower endpoint (pivot large)
    if np.isnan(hi_end) or np.isnan(lo_end):
        return -np.inf, np.inf, np.inf
    lo_end, hi_end = min(lo_end, hi_end), max(lo_end, hi_end)
    return lo_end, hi_end, hi_end - lo_end


def build_constraints(bpre, sel, c, G):
    """box selection on pre-coefficients; condition on signs of excluded cohorts."""
    cons = []
    for g in range(G):
        e = np.zeros(2 * G); e[g] = 1.0            # picks beta_pre_g  (pre block first)
        if sel[g]:
            cons.append((e.copy(), c))             #  beta_pre_g <= c
            cons.append((-e.copy(), c))            # -beta_pre_g <= c
        else:
            if bpre[g] > c:
                cons.append((-e.copy(), -c))       # beta_pre_g >= c
            else:
                cons.append((e.copy(), -c))        # beta_pre_g <= -c
    return cons


def run(s, delta_pre, c, rho, n_reps, theta_val=1.0):
    G = len(delta_pre)
    theta = np.full(G, theta_val)
    beta_pre = delta_pre.astype(float)             # delta_post = 0 -> all clean
    beta_post = theta.astype(float)                # beta_post = theta + 0
    # full 2G covariance: [pre block, post block], cohorts independent, within-cohort rho
    Sigma = np.zeros((2 * G, 2 * G))
    for g in range(G):
        Sigma[g, g] = s**2
        Sigma[G + g, G + g] = s**2
        Sigma[g, G + g] = rho * s**2
        Sigma[G + g, g] = rho * s**2
    Lchol = np.linalg.cholesky(Sigma)

    cn = cs = cl = 0
    ln = ls = ll = 0.0
    sel_inf_valid = 0
    for _ in range(n_reps):
        draw = Lchol @ rng.standard_normal(2 * G)
        bhat = np.concatenate([beta_pre, beta_post]) + draw
        bpre, bpost = bhat[:G], bhat[G:]
        sel = np.abs(bpre) <= c
        if sel.sum() == 0:
            sel = np.ones(G, bool)
        tgt = theta[sel].mean()                    # = 1 always (all clean)

        # naive
        m = bpost[sel].mean(); se = s / np.sqrt(sel.sum())
        cn += (m - z*se <= tgt <= m + z*se); ln += 2*z*se

        # sample-split (independent halves, each ~2x variance)
        d1 = Lchol @ rng.standard_normal(2 * G) * np.sqrt(2)
        d2 = Lchol @ rng.standard_normal(2 * G) * np.sqrt(2)
        h1 = np.concatenate([beta_pre, beta_post]) + d1
        h2 = np.concatenate([beta_pre, beta_post]) + d2
        s1 = np.abs(h1[:G]) <= c
        if s1.sum() == 0:
            s1 = np.ones(G, bool)
        m2 = h2[G:][s1].mean(); se2 = (s*np.sqrt(2)) / np.sqrt(s1.sum())
        cs += (h2[G:][s1].mean() is not None and (m2 - z*se2 <= theta[s1].mean() <= m2 + z*se2))
        ls += 2*z*se2

        # selective (exact polyhedral)
        eta = np.zeros(2 * G)
        eta[G + np.where(sel)[0]] = 1.0 / sel.sum()
        cons = build_constraints(bpre, sel, c, G)
        lo, hi, L = selective_interval(bhat, Sigma, eta, cons)
        cl += (lo <= tgt <= hi); ll += L
        sel_inf_valid += np.isfinite(L)

    N = n_reps
    return dict(
        cov_naive=100*cn/N, cov_split=100*cs/N, cov_sel=100*cl/N,
        len_naive=ln/N, len_split=ls/N, len_sel=ll/N,
        sel_finite=100*sel_inf_valid/N)


if __name__ == "__main__":
    G = 8
    c, rho = 0.40, 0.5
    # true pre-trends spread AROUND the threshold c=0.40 -> stochastic, asymmetric selection
    delta_pre = np.linspace(0.0, 0.60, G)
    print("Isolated Layer-1 race (all cohorts clean in post; target = 1 always)")
    print("Selection stochastic because true pre-trends straddle the threshold c=0.40\n")
    print(f"{'s':>6} | {'cov naive':>9} {'split':>6} {'selective':>9} | "
          f"{'len naive':>9} {'split':>7} {'selective':>9} | {'sel/split':>9}")
    for s in [0.35, 0.28, 0.22, 0.16, 0.11]:
        r = run(s, delta_pre, c, rho, n_reps=8000)
        print(f"{s:6.2f} | {r['cov_naive']:9.1f} {r['cov_split']:6.1f} {r['cov_sel']:9.1f} | "
              f"{r['len_naive']:9.4f} {r['len_split']:7.4f} {r['len_sel']:9.4f} | "
              f"{r['len_sel']/r['len_split']:9.2f}")
    print("\nRead: naive should UNDERcover (selection distortion). split & selective ~95%.")
    print("Layer 1 earns its keep only if selective length << split length at matched coverage.")
