"""
Section 3.4 figure (level band): honest coverage restored, the price in width, and
the breakdown curve. Three panels, saved as layer2_full.png.

Level bound Delta_Level(M)={|delta_post(e)|<=M}; estimator = raw post-average; the
FLCI half-length is cv(M/sigma)*sigma, and coverage of the causal target is nominal
iff M >= V (the true post-treatment violation). M and M* are in outcome units.
"""
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer2_full as L2

z = norm.ppf(0.975)
sv = 0.10                     # SE of the selected-aggregate post-average
theta = 1.0                   # a significant effect
rng = np.random.default_rng(3)

# ---- panel 1: coverage vs true violation V, point vs FLCI at two M ----
V_grid = np.linspace(0.0, 0.9, 25)
NR = 4000
def coverage(V, M):
    c_flci = c_pt = 0
    for _ in range(NR):
        est = theta + V + rng.normal(0, sv)          # flat pre; post shifted by V
        _, h = L2.flci_level(est, sv, M)
        c_flci += (est - h <= theta <= est + h)
        c_pt += (est - z*sv <= theta <= est + z*sv)  # naive point CI
    return 100*c_flci/NR, 100*c_pt/NR
cov_M3 = np.array([coverage(V, 0.30)[0] for V in V_grid])
cov_M6 = np.array([coverage(V, 0.60)[0] for V in V_grid])
cov_pt = np.array([coverage(V, 0.0)[1] for V in V_grid])

# ---- panel 2: FLCI half-width vs M (flat in V) ----
M_grid = np.linspace(0.0, 0.9, 60)
half = np.array([L2.flci_level(theta, sv, M)[1] for M in M_grid])

# ---- panel 3: breakdown curve at fixed data (est = 1) ----
est_fixed = 1.0
lo = np.array([est_fixed - L2.flci_level(est_fixed, sv, M)[1] for M in M_grid])
hi = np.array([est_fixed + L2.flci_level(est_fixed, sv, M)[1] for M in M_grid])
Mstar = M_grid[np.argmax(lo <= 0.0)] if np.any(lo <= 0.0) else np.nan
print(f"breakdown M* (outcome units) = {Mstar:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
fig.suptitle("Level-bound sensitivity: honest coverage, the price in width, and the breakdown curve", fontsize=11)

a = ax[0]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(V_grid, cov_M3, "o-", color="#2471a3", label="FLCI, M=0.30")
a.plot(V_grid, cov_M6, "s-", color="#16a085", label="FLCI, M=0.60")
a.plot(V_grid, cov_pt, "^-", color="#c0392b", label="naive point CI")
a.axvline(0.30, color="#2471a3", ls=":", lw=1); a.axvline(0.60, color="#16a085", ls=":", lw=1)
a.set_title("Coverage vs true violation V (honest iff M>=V)")
a.set_xlabel("true post-violation V (outcome units)"); a.set_ylabel("coverage (%)"); a.set_ylim(0, 102); a.legend(fontsize=7)

a = ax[1]
a.plot(M_grid, half, "-", color="#7d3c98", lw=2)
a.set_title("FLCI half-width vs assumed M\n(price of honesty, set by M)")
a.set_xlabel("assumed level bound M (outcome units)"); a.set_ylabel("half-width")

a = ax[2]
a.axhline(0, color="gray", lw=.8, ls=":")
a.fill_between(M_grid, lo, hi, color="#2471a3", alpha=.25)
a.plot(M_grid, lo, color="#2471a3"); a.plot(M_grid, hi, color="#2471a3")
a.axvline(Mstar, color="#c0392b", ls="--", lw=1.2, label=f"breakdown M*={Mstar:.2f}")
a.set_title("Breakdown curve at fixed data (effect=1)")
a.set_xlabel("assumed level bound M (outcome units)"); a.set_ylabel("LATT confidence band"); a.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig("layer2_full.png", dpi=130)
print("Saved -> layer2_full.png")
