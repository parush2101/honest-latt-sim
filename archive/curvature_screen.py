"""
Divergence cell: why the screen must be curvature, not level
------------------------------------------------------------
Constructs the case where level and curvature disagree, so the two screens
select opposite sets, and shows the curvature screen is the right one under the
SD(M) honest inference.

Three cohort types (true effect 1 for all; target = average post effect):
  - clean:  delta = 0
  - linear: delta_e = s*e            (large pre-level, ZERO curvature)
            -> benign under SD(M): the FLCI extrapolates the line and removes it,
               honest bias = 0.
  - curved: delta_e = (C/2) e^2      (SMALLER pre-level, curvature C)
            -> a threat under SD(M): curvature is the part the interval cannot
               absorb, honest bias = C * bbar1.

Level ranks the linear cohort as worst (biggest |pre| level); curvature ranks the
curved cohort as worst. So:
  - the LEVEL screen keeps {clean, curved}, drops the benign linear cohort
    -> it smuggles curvature into the aggregate AND needlessly loses scope;
  - the CURVATURE screen keeps {clean, linear}, drops the curved cohort
    -> the aggregate has no residual curvature, honestly covered at M ~ 0.
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

pre_e, post_e = L2.pre_e, L2.post_e
npre, npost = L2.npre, L2.npost
all_e = np.concatenate([pre_e, post_e])
V = L2.V_VEC
bbar1 = L2.max_bias_SD(V, 1.0)
z = norm.ppf(0.975)

S_LIN, C_CRV = 0.25, 0.08          # linear slope; curved curvature
c_level, c_curv = 0.80, 0.06       # thresholds (level scale; curvature scale)
N_CLEAN, N_LIN, N_CRV = 4, 3, 3
G = N_CLEAN + N_LIN + N_CRV
sigma, rho = 0.02, 0.5


def linear(s): return s * all_e
def curved(C): return 0.5 * C * (all_e ** 2)
def within_cov(sig, rr, n):
    idx = np.arange(n); R = rr ** np.abs(idx[:, None] - idx[None, :]); return (sig ** 2) * R


# cohort mean event studies: tau (=1 post) + differential trend
tau = np.concatenate([np.zeros(npre), np.ones(npost)])
deltas = [np.zeros(len(all_e))] * N_CLEAN + [linear(S_LIN)] * N_LIN + [curved(C_CRV)] * N_CRV
labels = ["clean"] * N_CLEAN + ["linear"] * N_LIN + ["curved"] * N_CRV
means = np.array([tau + d for d in deltas])
kind = np.array(labels)

# ---------- deterministic mechanism (no noise) ----------
print("=== archetypes: level vs curvature disagree ===")
print(f"{'type':>7} | {'pre max-level':>13} | {'curvature':>9} | {'honest bias |V.d|':>17}")
for lab, d in [("clean", np.zeros(len(all_e))), ("linear", linear(S_LIN)), ("curved", curved(C_CRV))]:
    lvl = np.max(np.abs(d[:npre])); crv = float(L2.max_abs_second_diff(d[None, :npre])[0]); bias = abs(V @ d)
    print(f"{lab:>7} | {lvl:13.3f} | {crv:9.3f} | {bias:17.3f}")

def agg_resid_curv(keep):
    dbar = np.mean([deltas[g] for g in range(G) if keep[g]], axis=0)
    return float(L2.max_abs_second_diff(dbar[None, :npre])[0])

keep_level_det = np.array([np.max(np.abs(deltas[g][:npre])) <= c_level for g in range(G)])
keep_curv_det = np.array([L2.max_abs_second_diff(deltas[g][None, :npre])[0] <= c_curv for g in range(G)])
print("\n=== who each screen keeps (noiseless) ===")
print(f"  LEVEL  screen (thr {c_level}): keeps {[labels[g] for g in range(G) if keep_level_det[g]]}")
print(f"      -> aggregate residual curvature = {agg_resid_curv(keep_level_det):.3f}  (breakdown M* ~ this)")
print(f"  CURV.  screen (thr {c_curv}): keeps {[labels[g] for g in range(G) if keep_curv_det[g]]}")
print(f"      -> aggregate residual curvature = {agg_resid_curv(keep_curv_det):.3f}  (breakdown M* ~ this)")

# ---------- coverage vs M under noise ----------
Sig = within_cov(sigma, rho, npre + npost)
Lc = np.linalg.cholesky(Sig)
rng = np.random.default_rng(7)
M_grid = np.linspace(0.0, 0.06, 25)
n_reps = 4000
TRUE = 1.0
cov_level = {M: 0 for M in M_grid}
cov_curv = {M: 0 for M in M_grid}
nsel_level = []; nsel_curv = []; lin_kept_curv = 0

for _ in range(n_reps):
    B = means + (Lc @ rng.standard_normal((npre + npost, G))).T
    lvl_stat = np.max(np.abs(B[:, :npre]), axis=1)
    crv_stat = L2.max_abs_second_diff(B[:, :npre])
    sel_L = lvl_stat <= c_level
    sel_C = crv_stat <= c_curv
    if sel_L.sum() == 0: sel_L = (lvl_stat == lvl_stat.min())
    if sel_C.sum() == 0: sel_C = (crv_stat == crv_stat.min())
    nsel_level.append(sel_L.sum()); nsel_curv.append(sel_C.sum())
    lin_kept_curv += np.sum(sel_C & (kind == "linear"))
    aggL = B[sel_L].mean(0); SigL = Sig / sel_L.sum()
    aggC = B[sel_C].mean(0); SigC = Sig / sel_C.sum()
    # max bias is linear in M (max_bias_SD(V,M) = M*bbar1), so avoid re-running the LP per M
    cenL = float(V @ aggL); svL = np.sqrt(V @ SigL @ V)
    cenC = float(V @ aggC); svC = np.sqrt(V @ SigC @ V)
    for M in M_grid:
        hL = L2.cv((M * bbar1) / svL) * svL; cov_level[M] += (cenL - hL <= TRUE <= cenL + hL)
        hC = L2.cv((M * bbar1) / svC) * svC; cov_curv[M] += (cenC - hC <= TRUE <= cenC + hC)

cvL = np.array([100 * cov_level[M] / n_reps for M in M_grid])
cvC = np.array([100 * cov_curv[M] / n_reps for M in M_grid])
def crossM(cov):
    ok = np.where(cov >= 95)[0]; return M_grid[ok[0]] if len(ok) else np.nan
print("\n=== coverage under noise ===")
print(f"  avg # kept: level={np.mean(nsel_level):.2f}/{G}, curvature={np.mean(nsel_curv):.2f}/{G}")
print(f"  linear cohorts kept by curvature screen: {lin_kept_curv/n_reps:.2f}/{N_LIN} (level screen drops them)")
print(f"  M for 95% coverage: level screen={crossM(cvL):.3f}, curvature screen={crossM(cvC):.3f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("Level and curvature disagree: the screen must match the SD(M) honesty check", fontsize=11)

a = ax[0]
pe = np.concatenate([pre_e, [0]])
a.axhline(0, color="gray", lw=.7, ls=":")
a.plot(pe, np.concatenate([linear(S_LIN)[:npre], [0]]), "o-", color="#c0392b",
       label=f"linear (level {np.max(np.abs(linear(S_LIN)[:npre])):.2f}, curvature 0): benign")
a.plot(pe, np.concatenate([curved(C_CRV)[:npre], [0]]), "s-", color="#2471a3",
       label=f"curved (level {np.max(np.abs(curved(C_CRV)[:npre])):.2f}, curvature {C_CRV}): threat")
a.set_title("Pre-trends: level flags the line, curvature flags the curve")
a.set_xlabel("event time (pre-period, reference e=0)"); a.set_ylabel("differential trend")
a.legend(fontsize=8, loc="upper left")

a = ax[1]
a.axhline(95, color="green", lw=.9, ls="--", label="95% nominal")
a.plot(M_grid, cvL, "o-", color="#c0392b", label="level screen (keeps curved -> needs large M)")
a.plot(M_grid, cvC, "s-", color="#2471a3", label="curvature screen (drops curved -> covers at M~0)")
a.set_title("Honest coverage of the LATT vs assumed bound M")
a.set_xlabel("smoothness bound M"); a.set_ylabel("coverage (%)"); a.set_ylim(0, 102)
a.legend(fontsize=8, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("curvature_screen.png", dpi=130)
print("\nSaved -> curvature_screen.png")
