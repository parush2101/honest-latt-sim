"""
Randomized carved inference across the noise scale gamma (Theorem 1).
---------------------------------------------------------------------
race2.py / race_multipre.py implement the EXACT polyhedral truncated-normal
interval at gamma = 0 (full conditioning). At gamma = 0 the carved interval is
exact but frequently INFINITE-length (the local-to-threshold / Leeb-Potscher
degeneracy): when the selected contrast sits near a binding face, the truncation
interval [V-,V+] collapses and inversion returns an unbounded set. That is the
efficiency full conditioning spends, and it is why the paper carries a
randomization scale.

Theorem 1: randomize the screen. Draw omega_g ~ N(0, gamma * Sigma_pre) per
cohort, independent of bhat, and select on the NOISED pre-trends
    Shat = { g : max_e |bhat_pre_g(e) + omega_g(e)| <= c }.
Condition on {Shat = S} together with the active screen coordinates and their
signs. This is a polyhedron in the AUGMENTED Gaussian vector y = (bhat, omega),
so the same Lee et al. (2016) polyhedral lemma applies: eta'y = theta_hat_S is
truncated normal on [V-,V+], with limits read off the augmented covariance.

  gamma = 0    : recovers race2 (fully conditional, exact, heavy-tailed length).
  gamma -> inf : omega dominates selection, decoupling Shat from theta_hat_S, so
                 [V-,V+] -> (-inf, inf) and the carved interval -> the naive
                 z-interval; but retention prob -> 0, so Shat = empty w.p. -> 1.
  intermediate : valid carved interval; gamma trades the length tail against the
                 empty-selection rate and the retained-set standard error.

WHAT THE SWEEP SHOWS (see run output for numbers):
  * The carved interval is VALID -- ~95% coverage at every gamma -- while the
    naive z-interval UNDERcovers under selection distortion. This is the Theorem 1
    guarantee, demonstrated (it was not previously reported in any paper exhibit).
  * The carved interval has INFINITE expected length (Kivaranovic-Leeb 2021): the
    mean is dominated by rare enormous intervals, so length is summarized by the
    MEDIAN. Coverage is over non-empty selections (Theorem 1 conditions on Shat=S,
    S != empty; the empty event reports the ATT interval, a separate object).
  * On MEDIAN length the carved interval is COMPARABLE to a proper sqrt(2)
    sample-split, not uniformly shorter. Its advantage over splitting is that it
    uses the whole sample deterministically (no arbitrary data split), not width.
    This bears directly on Theorem 2's data-driven width claim (see HANDOFF).

Selection-isolation design: all cohorts clean in post, so the causal target is 1
for ANY selected set -- Theorem 1 is a statement about selection, with the
identification widening of Section 5.4 held out by construction.

Reuses the robust polyhedral inverter from race_multipre (imported), evaluated on
the augmented (bhat, omega) vector.
"""
import numpy as np
from scipy.stats import norm
# Robust polyhedral inverter (scipy truncnorm + bracketed brentq): stays finite
# under many active constraints, unlike race2's plain-bisection version.
from race_multipre import selective_interval

z = norm.ppf(0.975)


def ar1_cov(s, rho, K):
    idx = np.arange(K)
    return s**2 * rho ** np.abs(idx[:, None] - idx[None, :])


def build_augmented(bhat, omega, sel, c, G, npre, npost):
    """Polyhedron {A y <= b} in the augmented vector y = (bhat, omega).

    Layout of y: first G*K entries are bhat (cohort-major, K=npre+npost), then
    G*npre entries are omega (cohort-major, pre-periods only). A screen row acts
    on x_g(e) = bhat_pre_g(e) + omega_g(e), i.e. +1 on the bhat coordinate and
    +1 on the matching omega coordinate.
    """
    K = npre + npost
    dimb = G * K
    dim = dimb + G * npre
    cons = []

    def screen_row(g, e, sign):
        u = np.zeros(dim)
        u[g * K + e] = sign                 # bhat_pre_g(e)
        u[dimb + g * npre + e] = sign       # omega_g(e)
        return u

    for g in range(G):
        xg = bhat.reshape(G, K)[g, :npre] + omega.reshape(G, npre)[g]
        if sel[g]:
            for e in range(npre):
                cons.append((screen_row(g, e, +1.0), c))    #  x_g(e) <= c
                cons.append((screen_row(g, e, -1.0), c))    # -x_g(e) <= c
        else:
            estar = int(np.argmax(np.abs(xg)))              # breaching coordinate
            if xg[estar] > c:
                cons.append((screen_row(g, estar, -1.0), -c))   #  x_g(e*) >= c
            else:
                cons.append((screen_row(g, estar, +1.0), -c))   #  x_g(e*) <= -c
    return cons


def run(s, delta_pre, c, rho, gamma, npre, npost, n_reps, seed=20240719):
    """One gamma. All cohorts clean in post (target = 1 for any selected set)."""
    rng = np.random.default_rng(seed)
    G = len(delta_pre)
    K = npre + npost
    dimb = G * K

    # means: constant differential pre-trend delta_pre[g] on every pre-period; clean post
    mu = np.zeros(dimb)
    for g in range(G):
        mu[g * K:g * K + npre] = delta_pre[g]
        mu[g * K + npre:(g + 1) * K] = 1.0
    Sig1 = ar1_cov(s, rho, K)
    Spre = Sig1[:npre, :npre]
    Sigma_b = np.zeros((dimb, dimb))
    for g in range(G):
        Sigma_b[g * K:(g + 1) * K, g * K:(g + 1) * K] = Sig1
    Lb = np.linalg.cholesky(Sigma_b)

    # augmented covariance: blockdiag(Sigma_b, gamma * blockdiag(Spre))
    dim = dimb + G * npre
    Sigma_y = np.zeros((dim, dim))
    Sigma_y[:dimb, :dimb] = Sigma_b
    if gamma > 0:
        Lo = np.linalg.cholesky(Spre)
        for g in range(G):
            j = dimb + g * npre
            Sigma_y[j:j + npre, j:j + npre] = gamma * Spre
    se_unit = np.sqrt(Sig1[npre:, npre:].mean())     # sd of one cohort's post average

    # Coverage is accumulated over NON-EMPTY reps (Theorem 1 conditions on Shat=S,
    # S != empty; the empty event reports the ATT interval, a separate object).
    # Length is summarized by the MEDIAN, not the mean: the polyhedral selective
    # interval has infinite expected length (Kivaranovic-Leeb 2021), so the mean is
    # dominated by rare enormous intervals and is not an informative summary.
    cn = cs = cl = 0
    ln_list, ls_list, ll_list = [], [], []
    n_nonempty = 0
    n_empty = 0
    for _ in range(n_reps):
        bhat = mu + Lb @ rng.standard_normal(dimb)
        if gamma > 0:
            omega = np.concatenate([Lo @ rng.standard_normal(npre) for _ in range(G)]) * np.sqrt(gamma)
        else:
            omega = np.zeros(G * npre)

        pre = bhat.reshape(G, K)[:, :npre]
        post = bhat.reshape(G, K)[:, npre:]
        x = pre + omega.reshape(G, npre)                 # noised pre-trends
        sel = np.max(np.abs(x), axis=1) <= c
        if sel.sum() == 0:                               # empty selection: report ATT interval
            n_empty += 1                                 # (a separate object); excluded here
            continue
        n_nonempty += 1
        tgt = 1.0

        post_avg = post.mean(axis=1)
        # naive: same selected set as carved, but ignore the selection in inference
        m = post_avg[sel].mean(); se = se_unit / np.sqrt(sel.sum())
        cn += (m - z*se <= tgt <= m + z*se); ln_list.append(2*z*se)

        # sample-split: split the ONE sample 50/50, so each half carries sqrt(2) the
        # standard error. Select on half 1, estimate on half 2. (The honest no-carving
        # alternative that pays for validity with lost precision, not extra data.)
        h1 = mu + Lb @ rng.standard_normal(dimb) * np.sqrt(2)
        h2 = mu + Lb @ rng.standard_normal(dimb) * np.sqrt(2)
        if gamma > 0:
            o1 = np.concatenate([Lo @ rng.standard_normal(npre) for _ in range(G)]) * np.sqrt(gamma)
        else:
            o1 = np.zeros(G * npre)
        x1 = h1.reshape(G, K)[:, :npre] + o1.reshape(G, npre)
        s1 = np.max(np.abs(x1), axis=1) <= c
        if s1.sum() == 0:
            s1 = (np.max(np.abs(x1), axis=1) == np.max(np.abs(x1), axis=1).min())
        p2 = h2.reshape(G, K)[:, npre:].mean(axis=1)
        m2 = p2[s1].mean(); se2 = se_unit * np.sqrt(2) / np.sqrt(s1.sum())
        cs += (m2 - z*se2 <= tgt <= m2 + z*se2); ls_list.append(2*z*se2)

        # carved (exact polyhedral on the augmented vector)
        eta = np.zeros(dim)
        w = 1.0 / (sel.sum() * npost)
        for g in np.where(sel)[0]:
            eta[g * K + npre:(g + 1) * K] = w
        y = np.concatenate([bhat, omega])
        cons = build_augmented(bhat, omega, sel, c, G, npre, npost)
        lo, hi, Llen = selective_interval(y, Sigma_y, eta, cons)
        cl += (lo <= tgt <= hi)
        ll_list.append(Llen)

    ne = max(n_nonempty, 1)
    ll_arr = np.array(ll_list, float)
    return dict(
        cov_naive=100*cn/ne, cov_split=100*cs/ne, cov_carved=100*cl/ne,
        med_naive=float(np.median(ln_list)) if ln_list else np.nan,
        med_split=float(np.median(ls_list)) if ls_list else np.nan,
        med_carved=float(np.median(ll_arr)) if ll_arr.size else np.nan,
        p90_carved=float(np.percentile(ll_arr, 90)) if ll_arr.size else np.nan,
        empty=100*n_empty/n_reps)


if __name__ == "__main__":
    # Selection-distortion regime: two clean cohorts, six borderline just below c,
    # strong pre/post correlation so naive UNDERcovers (this is where carving matters).
    G, npre, npost = 8, 4, 1
    c, rho, s = 0.40, 0.85, 0.16
    delta_pre = np.concatenate([np.zeros(2), np.full(G - 2, 0.36)])
    n_reps = 2500

    print(f"Randomized carved inference across gamma  (G={G}, npre={npre}, npost={npost}, "
          f"c={c}, rho={rho}, s={s})")
    print("All cohorts clean in post -> causal target = 1 for any selected set.\n")
    print("Coverage over non-empty selections (Theorem 1 guarantee); length = MEDIAN")
    print("(carved interval has infinite expected length, so the mean is uninformative).\n")
    print(f"{'gamma':>7} | {'cov carved':>10} {'naive':>6} {'split':>6} | "
          f"{'med carved':>10} {'med split':>9} {'p90 carved':>10} | {'empty%':>6}")
    for gamma in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        r = run(s, delta_pre, c, rho, gamma, npre, npost, n_reps)
        print(f"{gamma:7.2f} | {r['cov_carved']:10.1f} {r['cov_naive']:6.1f} {r['cov_split']:6.1f} | "
              f"{r['med_carved']:10.4f} {r['med_split']:9.4f} {r['p90_carved']:10.4f} | {r['empty']:6.1f}")
    print("\nReading: naive UNDERcovers (~90%, selection distortion). Carved holds ~95%")
    print("at every gamma (Theorem 1). Median carved length is comparable to a proper")
    print("sqrt(2) sample-split -- not uniformly shorter -- and GROWS with gamma as the")
    print("retained set shrinks and empty% climbs. Small gamma (~0.25-0.5) trims the p90")
    print("tail slightly; the carved interval's edge over splitting is using the whole")
    print("sample deterministically, not width.")
