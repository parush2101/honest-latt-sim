"""
CHC application, sensitivity/breakdown analysis.

Reports the object the paper argues a practitioner should report (Sec 4.4): the
SD(M) FLCI for the credible-subpopulation LATT as the assumed smoothness bound M
is relaxed, and the breakdown value M* at which the band first admits zero.

Estimator matches chc_figure.py: long differences in log elderly mortality against
the never-treated counties, baseline year g-1 (so e=-1 is normalized to 0).
Covariance comes from a stratified county-level cluster bootstrap, which is the
honest covariance for the estimator actually used.

Two SEs are reported, mirroring the panel simulation of Sec 4.6:
  - fixed-set:  credible cohorts held at their full-sample selection
  - re-select:  cohorts re-selected inside each bootstrap draw
The gap is the variance component induced by the randomness of selection itself.
"""
import numpy as np, pandas as pd, json
from collections import defaultdict
from scipy.stats import norm
from scipy.optimize import brentq, linprog

# ---------------- event-time grid ----------------
REF_E = -1                       # baseline year g-1 -> delta_{-1} = 0 by construction
PRE_E = np.arange(-6, -1)        # -6..-2  (used to fit the extrapolated trend)
POST_E = np.arange(1, 6)         # 1..5    (target: average effect over first five years)
ALL_E = np.arange(-6, 6)         # contiguous grid; e=0 carries zero weight (transition year)
C_SEL = 0.006                    # credibility threshold, as in chc_figure.py

# ---------------- FLCI machinery on this grid ----------------
def cv(t, alpha=0.05):
    """(1-alpha) quantile of |N(t,1)|."""
    t = abs(t)
    f = lambda q: (norm.cdf(q - t) - norm.cdf(-q - t)) - (1 - alpha)
    return brentq(f, 0.0, t + 10.0)

def extrapolation_weights():
    """theta_hat = mean(beta_post) - mean(linear extrapolation of beta_pre to post).
    Returns v aligned to ALL_E so that E[theta_hat] - theta = v'delta."""
    X = np.column_stack([np.ones(len(PRE_E)), PRE_E])
    Xpost = np.column_stack([np.ones(len(POST_E)), POST_E])
    H = Xpost @ np.linalg.inv(X.T @ X) @ X.T          # beta_pre -> extrapolated post
    l_post = np.ones(len(POST_E)) / len(POST_E)
    w_pre = -(H.T @ l_post)
    v = np.zeros(len(ALL_E))
    idx = {e: i for i, e in enumerate(ALL_E)}
    for j, e in enumerate(PRE_E):
        v[idx[e]] = w_pre[j]
    for j, e in enumerate(POST_E):
        v[idx[e]] = l_post[j]
    return v

V_VEC = extrapolation_weights()

def max_bias_SD(v, M):
    """max_{delta in SD(M)} |v'delta| with delta_{REF_E}=0. LP; +inf if unbounded."""
    idx = {e: i for i, e in enumerate(ALL_E)}
    n = len(ALL_E)
    A = []
    for e in ALL_E:
        if (e - 1) in idx and (e + 1) in idx:
            row = np.zeros(n)
            row[idx[e - 1]] += 1; row[idx[e]] += -2; row[idx[e + 1]] += 1
            A.append(row)
    A = np.array(A)
    A_ub = np.vstack([A, -A]); b_ub = np.full(2 * len(A), M)
    A_eq = np.zeros((1, n)); A_eq[0, idx[REF_E]] = 1.0
    kw = dict(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0],
              bounds=[(None, None)] * n, method="highs")
    hi = linprog(-v, **kw); lo = linprog(v, **kw)
    if hi.status == 3 or lo.status == 3:
        return np.inf
    return max(abs(-hi.fun), abs(lo.fun))

def flci(bhat_full, sigma_v, M, alpha=0.05):
    """bhat_full aligned to ALL_E. Returns (center, half_length) of the SD(M) FLCI."""
    center = float(V_VEC @ bhat_full)
    bbar = max_bias_SD(V_VEC, M)
    if not np.isfinite(bbar):
        return center, np.inf
    return center, cv(bbar / sigma_v, alpha) * sigma_v

# ---------------- data ----------------
df = pd.read_csv("/tmp/chc_slim.csv")
Y, POP = {}, {}
for r in df.itertuples():
    Y[(int(r.fips), int(r.year))] = np.log(r.amr_eld)
    POP[(int(r.fips), int(r.year))] = r.copop
ft = (df.dropna(subset=['chc_year_exp']).drop_duplicates('fips')
        .set_index('fips')['chc_year_exp'].astype(int).to_dict())
never = [f for f in df.fips.unique() if f not in ft]
byg = defaultdict(list)
for f, g in ft.items():
    byg[g].append(f)
EARLY = sorted([g for g in byg if 1965 <= g <= 1974])
slopes_full = {int(k): v for k, v in json.load(open("/tmp/chc_slopes.json")).items()}
CREDIBLE = [g for g in EARLY if abs(slopes_full.get(g, 9)) < C_SEL]

def event_study(cohorts, nev, byg_s, w=False):
    """Weighted event study over `cohorts` vs never-treated `nev`, aligned to ALL_E."""
    nm_cache = {}
    def nm(y):
        if y in nm_cache: return nm_cache[y]
        v = [(Y[(f, y)], POP.get((f, y), 1) if w else 1) for f in nev if (f, y) in Y]
        if not v:
            nm_cache[y] = np.nan; return np.nan
        a = np.array([x[0] for x in v]); wt = np.array([x[1] for x in v])
        nm_cache[y] = np.sum(a * wt) / wt.sum(); return nm_cache[y]
    out = np.full(len(ALL_E), np.nan)
    for i, e in enumerate(ALL_E):
        num = den = 0.0
        for g in cohorts:
            m = [(Y[(f, g + e)] - Y[(f, g - 1)], POP.get((f, g - 1), 1) if w else 1)
                 for f in byg_s[g] if (f, g + e) in Y and (f, g - 1) in Y]
            if not m: continue
            a = np.array([x[0] for x in m]); wt = np.array([x[1] for x in m])
            b0, b1 = nm(g + e), nm(g - 1)
            if not (np.isfinite(b0) and np.isfinite(b1)): continue
            dt = np.sum(a * wt) / wt.sum() - (b0 - b1)
            wg = sum(POP.get((f, g - 1), 1) if w else 1 for f in byg_s[g])
            num += wg * dt; den += wg
        out[i] = 100 * num / den if den else np.nan
    return out

def pre_slope(cohort_es):
    """OLS slope of the pre-period path, for the credibility screen (per-year, log units)."""
    ok = [(e, cohort_es[i]) for i, e in enumerate(ALL_E) if e in PRE_E and np.isfinite(cohort_es[i])]
    if len(ok) < 3: return np.inf
    e = np.array([x[0] for x in ok]); y = np.array([x[1] for x in ok]) / 100.0
    return np.polyfit(e, y, 1)[0]

# ---------------- point estimates ----------------
beta_latt = event_study(CREDIBLE, never, byg)
beta_att = event_study(EARLY, never, byg)
print(f"cohorts 1965-74: {EARLY}")
print(f"credible (|slope| < {C_SEL}): {CREDIBLE}   [dropped: {sorted(set(EARLY)-set(CREDIBLE))}]")
print(f"\navg post e1-5   ATT  = {np.nanmean([beta_att[i] for i,e in enumerate(ALL_E) if e in POST_E]):+.2f}%")
print(f"avg post e1-5   LATT = {np.nanmean([beta_latt[i] for i,e in enumerate(ALL_E) if e in POST_E]):+.2f}%")
print(f"LATT extrapolation-adjusted center (v'beta) = {float(V_VEC@np.nan_to_num(beta_latt)):+.3f}%")

# ---------------- cluster bootstrap ----------------
def bootstrap(n_boot=400, seed=7, reselect=False):
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n_boot):
        byg_b = {g: list(rng.choice(byg[g], size=len(byg[g]), replace=True)) for g in EARLY}
        nev_b = list(rng.choice(never, size=len(never), replace=True))
        if reselect:
            sel = [g for g in EARLY
                   if abs(pre_slope(event_study([g], nev_b, byg_b))) < C_SEL]
            if not sel: sel = EARLY
        else:
            sel = CREDIBLE
        b = event_study(sel, nev_b, byg_b)
        centers.append(float(V_VEC @ np.nan_to_num(b)))
    return np.array(centers)

print("\nbootstrapping (fixed set)...")
c_fix = bootstrap(reselect=False)
print("bootstrapping (re-selecting)...")
c_res = bootstrap(reselect=True)
se_fix, se_res = c_fix.std(ddof=1), c_res.std(ddof=1)
print(f"\nSE of LATT center: fixed-set = {se_fix:.3f}   re-select = {se_res:.3f}"
      f"   (selection-induced component = {se_res-se_fix:+.3f})")

# ---------------- breakdown curve ----------------
print("\nSD(M) FLCI for the credible-subpopulation LATT (percent change in elderly mortality)")
print(f"{'M':>7} {'center':>9} {'lower':>9} {'upper':>9}   admits 0?")
M_GRID = np.round(np.arange(0.0, 0.301, 0.005), 4)
rows, Mstar = [], None
for M in M_GRID:
    ctr, half = flci(np.nan_to_num(beta_latt), se_res, M)
    lo, hi = ctr - half, ctr + half
    rows.append((M, ctr, lo, hi))
    if Mstar is None and lo <= 0.0 <= hi:
        Mstar = M
for M, ctr, lo, hi in rows:
    if abs(M * 200 - round(M * 200)) < 1e-9 and round(M * 1000) % 25 == 0:
        print(f"{M:7.3f} {ctr:+9.3f} {lo:+9.3f} {hi:+9.3f}   {'YES' if lo <= 0 <= hi else 'no'}")
print(f"\nbreakdown value  M* = {Mstar:.3f}"
      if Mstar is not None else "\nno breakdown on grid")

np.save("/tmp/chc_breakdown_rows.npy", np.array(rows))
json.dump({"Mstar": float(Mstar) if Mstar is not None else None,
           "se_fix": float(se_fix), "se_res": float(se_res),
           "credible": CREDIBLE, "early": EARLY,
           "beta_latt": beta_latt.tolist(), "beta_att": beta_att.tolist()},
          open("/tmp/chc_breakdown.json", "w"), indent=1)
print("wrote /tmp/chc_breakdown.json")
