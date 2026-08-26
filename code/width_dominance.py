"""
Section 4 figure (width dominance): a direct check of Theorem~\\ref{thm:width} and the
explicit equal-weights threshold. When the dropped cohorts' violation is large enough,
the honest credible-subpopulation (LATT) interval is strictly shorter than the honest
ATT (Rambachan-Roth) interval---and BOTH cover their targets, so the gain is not bought
by undercoverage.

Design (Cor. "Explicit threshold under equal weights", ex-ante / informative limit):
  K equally weighted cohorts, each an independent post-treatment aggregate
  b_g ~ N(tau_g + V_g, sigma^2), tau_g = 1 (so ATT = LATT = 1, isolating the violation).
  k clean cohorts (V_g=0) are retained; K-k confounded cohorts (V_g=V) are dropped.
  Ex-ante selection S = clean set, so both intervals are ordinary level-bound FLCIs
  (no carving needed) and the width comparison is exact.

Level-bound FLCI half-width  h(B,s) = cv_alpha(B/s) * s   [L2.cv is the folded-normal cv].
  LATT: retained set truly flat, B_S = 0,  s_S = sigma/sqrt(k)  ->  h_S = z * sigma/sqrt(k).
  ATT : must bound the full-set violation, B_G = f V (f=(K-k)/K),  s_G = sigma/sqrt(K)
        ->  h_G = cv(fV / s_G) * s_G,   rising in V.
  Crossing  h_S = h_G  at  V* = sigma * cv^{-1}( z sqrt(K/k) ) / (f sqrt(K)).
  At K=2, k=1, alpha=.05:  V* = 1.59 sigma.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2                      # reuse cv_alpha (folded-normal critical value)

z = norm.ppf(0.975)
cv = L2.cv                                    # cv(t) = (1-alpha) quantile of |N(t,1)|


# ---------- honest FLCI half-widths (deterministic: fixed-length) ----------
def h_latt(sigma, k):
    """Retained set truly flat (B_S=0): half-width = z * sigma/sqrt(k)."""
    return cv(0.0) * (sigma / np.sqrt(k))

def h_att(sigma, K, k, V):
    """ATT interval bounds the full-set violation B_G = f V, s_G = sigma/sqrt(K)."""
    f = (K - k) / K
    sG = sigma / np.sqrt(K)
    return cv((f * V) / sG) * sG

def Vstar(sigma, K, k):
    """Violation at which the two honest half-widths are equal."""
    f = (K - k) / K
    # solve cv(t) = z*sqrt(K/k) for t, then V* = sigma * t / (f*sqrt(K))
    target = z * np.sqrt(K / k)
    t = brentq(lambda tt: cv(tt) - target, 0.0, 50.0)
    return sigma * t / (f * np.sqrt(K))


# ---------- Monte Carlo: both intervals honest; naive ATT collapses ----------
def coverage(sigma, K, k, V, n_reps=40000, seed=7):
    """Ex-ante S = clean set. Returns coverage (%) of LATT-FLCI, ATT-FLCI, naive-ATT CI,
    all for the true target 1.0 (tau homogeneous => ATT=LATT=1)."""
    rng = np.random.default_rng(seed)
    f = (K - k) / K
    sS, sG = sigma / np.sqrt(k), sigma / np.sqrt(K)
    hS, hG = h_latt(sigma, k), h_att(sigma, K, k, V)
    # cohort means: k clean at 1, K-k confounded at 1+V
    mu = np.concatenate([np.ones(k), np.full(K - k, 1.0 + V)])
    cL = cG = cNaive = 0
    for _ in range(n_reps):
        b = rng.normal(mu, sigma)
        latt = b[:k].mean()                       # retained (clean) aggregate
        att = b.mean()                            # full aggregate, biased by f*V
        cL += (latt - hS <= 1.0 <= latt + hS)     # LATT honest FLCI (B_S=0)
        cG += (att - hG <= 1.0 <= att + hG)       # ATT honest FLCI (B_G=fV)
        cNaive += (att - z * sG <= 1.0 <= att + z * sG)   # naive ATT CI (no bound)
    N = n_reps
    return 100 * cL / N, 100 * cG / N, 100 * cNaive / N


if __name__ == "__main__":
    sigma = 1.0                                   # widths reported in units of sigma
    print("Width dominance: honest LATT interval vs honest ATT interval\n")
    print(f"  cv_.95(0) = z = {cv(0.0):.4f}\n")
    print(f"  {'(K,k)':>7} {'f':>5} {'V*/sigma':>9}   (violation at which LATT interval becomes shorter)")
    for K, k in [(2, 1), (4, 2), (6, 3), (6, 4), (12, 6)]:
        print(f"  {f'({K},{k})':>7} {(K-k)/K:5.2f} {Vstar(sigma, K, k):9.3f}")

    # ================= FIGURE =================
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.suptitle("Width dominance: the honest LATT interval is shorter than the honest ATT "
                 "interval once the dropped violation exceeds $V^\\ast$ (both cover)", fontsize=10.5)

    # ---- panel (a): half-widths vs V/sigma, K=2,k=1, crossing at V* ----
    K, k = 2, 1
    Vs = np.linspace(0.0, 3.0, 200)
    hS = np.full_like(Vs, h_latt(sigma, k))
    hG = np.array([h_att(sigma, K, k, V) for V in Vs])
    hNaive = np.full_like(Vs, z * sigma / np.sqrt(K))     # naive ATT half-width (dishonest ref)
    vstar = Vstar(sigma, K, k)
    a = ax[0]
    a.plot(Vs, hG, color="#c0392b", lw=2, label="ATT honest interval $h_G$")
    a.plot(Vs, hS, color="#2471a3", lw=2, label="LATT honest interval $h_S$")
    a.plot(Vs, hNaive, color="gray", lw=1, ls=":", label="naive ATT (no bound)")
    a.axvline(vstar, color="black", ls="--", lw=1.1)
    a.fill_between(Vs, 0, np.max(hG), where=(Vs >= vstar), color="#2471a3", alpha=0.07)
    a.text(vstar + 0.05, 0.4, f"$V^\\ast={vstar:.2f}\\,\\sigma$", fontsize=9)
    a.text(2.15, h_latt(sigma, k) + 0.15, "LATT shorter", color="#2471a3", fontsize=8.5)
    a.set_title("(a) Half-widths, $K=2,\\ k=1$")
    a.set_xlabel("dropped violation $V/\\sigma$"); a.set_ylabel("interval half-width $/\\sigma$")
    a.set_ylim(0, np.max(hG)); a.legend(fontsize=7.5, loc="upper left")

    # ---- panel (b): Monte Carlo coverage, K=2,k=1 ----
    Vg = np.linspace(0.0, 3.0, 25)
    covs = np.array([coverage(sigma, K, k, V) for V in Vg])
    a = ax[1]
    a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
    a.plot(Vg, covs[:, 1], "s-", color="#c0392b", ms=4, label="ATT honest FLCI")
    a.plot(Vg, covs[:, 0], "o-", color="#2471a3", ms=4, label="LATT honest FLCI")
    a.plot(Vg, covs[:, 2], "^-", color="gray", ms=4, label="naive ATT CI")
    a.axvline(vstar, color="black", ls="--", lw=1.1)
    a.set_title("(b) Coverage of the target ($=1$)")
    a.set_xlabel("dropped violation $V/\\sigma$"); a.set_ylabel("coverage (%)")
    a.set_ylim(0, 102); a.legend(fontsize=7.5, loc="lower left")

    # ---- panel (c): V*/sigma vs dropped fraction, several K ----
    a = ax[2]
    for K in [2, 4, 8]:
        ks = np.arange(1, K)                       # 1..K-1 retained
        vv = [Vstar(sigma, K, kk) for kk in ks]
        frac = [(K - kk) / K for kk in ks]         # dropped fraction f
        a.plot(frac, vv, "o-", ms=4, lw=1.4, label=f"$K={K}$")
    a.axhline(np.sqrt(2), color="gray", ls=":", lw=1)
    a.text(0.02, np.sqrt(2) + 0.05, "$\\sqrt{2}\\,\\sigma$ (MSE threshold)", fontsize=7.5, color="gray")
    a.set_title("(c) Threshold $V^\\ast$ across designs")
    a.set_xlabel("dropped fraction $f=(K-k)/K$"); a.set_ylabel("$V^\\ast/\\sigma$")
    a.legend(fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("width_dominance.png", dpi=130)
    print("\nSaved -> width_dominance.png")

    # ---- console verification at, below, and above V* (K=2,k=1) ----
    K, k = 2, 1
    vstar = Vstar(sigma, K, k)
    print(f"\nVerification (K=2,k=1, V*={vstar:.3f} sigma):")
    print(f"  {'V/sigma':>8} {'h_S':>7} {'h_G':>7} {'shorter':>9} {'cov_LATT':>9} {'cov_ATT':>8} {'cov_naive':>10}")
    for V in [0.8, vstar, 2.0, 2.5]:
        hS_, hG_ = h_latt(sigma, k), h_att(sigma, K, k, V)
        cL, cG, cN = coverage(sigma, K, k, V)
        who = "LATT" if hS_ < hG_ else "ATT"
        print(f"  {V:8.2f} {hS_:7.3f} {hG_:7.3f} {who:>9} {cL:8.1f}% {cG:7.1f}% {cN:9.1f}%")
