"""
Layer 2 prototype: RR-style Delta bounds on the SELECTED cohorts' delta_post,
built on the same master-axis DGP.

Causal target (credible subpopulation):  theta_LATT = mean_{g in S} theta_g = 1.
We observe  beta_hat_post,g = theta_g + delta_post,g  and  beta_hat_pre,g = delta_pre,g.
Point estimate P = mean_{g in S} beta_hat_post,g is biased by mean delta_post over S.

We form an honest (conservative, RR-flavoured) interval for theta_LATT under a
restriction on delta_post.  Two flavours:

  (1) Relative-magnitudes  RM(Mbar):  |delta_post,g| <= Mbar * |delta_pre,g|
      -> identification half-width = Mbar * mean_{g in S}|beta_hat_pre,g|
      (the bound is ANCHORED to the observed pre-trends)

  (2) Absolute / "from theory"  ABS(B): |delta_post,g| <= B
      -> identification half-width = B     (bound is EXTERNAL, not from the data)

Interval = [ P - (id half-width) - z*se ,  P + (id half-width) + z*se ],
where se = s/sqrt(|S|) is the sampling se of P.  (Prototype: a transparent
conservative band, not the full ARP/FLCI machinery.)

Question: does RM inherit the informativeness scope (fail when pre-trends are
uninformative), while ABS escapes it at the cost of needing external knowledge?
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(99)

N_clean, N_conf = 6, 6
G = N_clean + N_conf
theta = np.ones(G)
V = 0.6
s = 0.20
rho_noise = 0.3
c = 0.40
n_reps = 8000
z = norm.ppf(0.975)
cov = np.array([[s**2, rho_noise*s*s], [rho_noise*s*s, s**2]])
L = np.linalg.cholesky(cov)
clean_mask = np.array([True]*N_clean + [False]*N_conf)
TRUE = 1.0

Mbar = 1.0            # relative-magnitudes bound
B = 0.6              # absolute bound "from theory" (== V, i.e. theory is correct)
B_wrong = 0.3        # absolute bound if theory UNDER-states the violation

info_grid = np.linspace(0.0, 1.5, 25)

cov_naive, cov_rm, cov_abs, cov_abs_wrong = [], [], [], []
hw_rm, hw_abs = [], []

for info in info_grid:
    delta_pre = np.where(clean_mask, 0.0, info*V)
    delta_post = np.where(clean_mask, 0.0, V)
    beta_pre = delta_pre
    beta_post = theta + delta_post

    cn = np.zeros(n_reps, bool); crm = np.zeros(n_reps, bool)
    cab = np.zeros(n_reps, bool); cabw = np.zeros(n_reps, bool)
    hrm = np.empty(n_reps); hab = np.empty(n_reps)

    for r in range(n_reps):
        bh = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T
        sel = np.abs(bh[:, 0]) <= c
        if sel.sum() == 0:
            sel = np.ones(G, bool)
        P = bh[:, 1][sel].mean()
        se = s/np.sqrt(sel.sum())
        tgt = theta[sel].mean()                       # = 1

        # naive point interval (no identification allowance)
        cn[r] = (P - z*se <= tgt <= P + z*se)
        # RM: id half-width anchored to observed pre-trends of the selected set
        idhw_rm = Mbar * np.abs(bh[:, 0][sel]).mean()
        hrm[r] = idhw_rm
        crm[r] = (P - idhw_rm - z*se <= tgt <= P + idhw_rm + z*se)
        # ABS: external bound
        hab[r] = B
        cab[r] = (P - B - z*se <= tgt <= P + B + z*se)
        cabw[r] = (P - B_wrong - z*se <= tgt <= P + B_wrong + z*se)

    cov_naive.append(cn.mean()*100); cov_rm.append(crm.mean()*100)
    cov_abs.append(cab.mean()*100); cov_abs_wrong.append(cabw.mean()*100)
    hw_rm.append(hrm.mean()); hw_abs.append(hab.mean())

cov_naive = np.array(cov_naive); cov_rm = np.array(cov_rm)
cov_abs = np.array(cov_abs); cov_abs_wrong = np.array(cov_abs_wrong)
hw_rm = np.array(hw_rm); hw_abs = np.array(hw_abs)

print(f"{'info':>5} | {'naive%':>7} {'RM%':>7} {'ABS%':>7} {'ABSwrong%':>9} | "
      f"{'RM halfw':>8} {'ABS halfw':>9}")
for i in range(0, len(info_grid), 2):
    print(f"{info_grid[i]:5.2f} | {cov_naive[i]:7.1f} {cov_rm[i]:7.1f} {cov_abs[i]:7.1f} "
          f"{cov_abs_wrong[i]:9.1f} | {hw_rm[i]:8.3f} {hw_abs[i]:9.3f}")

# ---------- sensitivity / breakdown curves at two info values ----------
def expected_band(info, Mgrid):
    delta_pre = np.where(clean_mask, 0.0, info*V)
    delta_post = np.where(clean_mask, 0.0, V)
    beta_pre = delta_pre; beta_post = theta + delta_post
    Ps, Ds, ses = [], [], []
    for r in range(4000):
        bh = np.column_stack([beta_pre, beta_post]) + (L @ rng.standard_normal((2, G))).T
        sel = np.abs(bh[:, 0]) <= c
        if sel.sum() == 0: sel = np.ones(G, bool)
        Ps.append(bh[:, 1][sel].mean())
        Ds.append(np.abs(bh[:, 0][sel]).mean())
        ses.append(s/np.sqrt(sel.sum()))
    P = np.mean(Ps); D = np.mean(Ds); se = np.mean(ses)
    lb = P - Mgrid*D - z*se
    ub = P + Mgrid*D + z*se
    return P, lb, ub

Mgrid = np.linspace(0, 3, 60)

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.5))
fig.suptitle("Layer 2: RR-style Delta bounds on the selected set\n"
             "relative-magnitudes bound INHERITS the informativeness scope; "
             "an external/theory bound ESCAPES it", fontsize=11)

a = ax[0, 0]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(info_grid, cov_naive, "o-", color="#e67e22", label="naive point CI")
a.plot(info_grid, cov_rm, "s-", color="#2471a3", label=f"RM(Mbar={Mbar})")
a.plot(info_grid, cov_abs, "^-", color="#27ae60", label=f"ABS(B={B}) [theory correct]")
a.plot(info_grid, cov_abs_wrong, "v--", color="#c0392b", label=f"ABS(B={B_wrong}) [theory too small]")
a.set_title("Causal coverage of theta_LATT")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("coverage (%)")
a.set_ylim(0, 102); a.legend(fontsize=7)

a = ax[0, 1]
a.plot(info_grid, hw_rm, "s-", color="#2471a3", label=f"RM(Mbar={Mbar}) half-width")
a.plot(info_grid, hw_abs, "^-", color="#27ae60", label=f"ABS(B={B}) half-width")
a.set_title("Identification half-width\n(RM collapses to ~0 exactly where it is WRONG)")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("half-width")
a.legend(fontsize=8)

for (a, info_val, ttl) in [(ax[1, 0], 0.5, "Sensitivity curve, LOW info=0.5"),
                           (ax[1, 1], 1.2, "Sensitivity curve, HIGH info=1.2")]:
    P, lb, ub = expected_band(info_val, Mgrid)
    a.fill_between(Mgrid, lb, ub, alpha=.25, color="#2471a3", label="RM identified band (±CI)")
    a.axhline(TRUE, color="black", ls="-", lw=1.2, label="true LATT = 1")
    a.axhline(P, color="#2471a3", ls=":", lw=1, label="point estimate P")
    # breakdown Mbar: first Mbar where band covers truth
    covers = (lb <= TRUE) & (TRUE <= ub)
    if covers.any():
        mstar = Mgrid[np.argmax(covers)]
        a.axvline(mstar, color="#c0392b", ls="--", lw=1, label=f"breakdown Mbar*={mstar:.2f}")
    a.set_title(ttl)
    a.set_xlabel("Mbar (relative-magnitudes bound)"); a.set_ylabel("theta_LATT")
    a.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("layer2.png", dpi=130)
print("\nSaved figure -> layer2.png")
