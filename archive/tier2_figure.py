"""Tier 2 figure: does the Tier 1 spine survive real panel estimation + estimated Sigma_hat?"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tier2

grid = np.linspace(0.0, 1.6, 13)
cs_b, la_b, cov, din, se_est, se_mc = [], [], [], [], [], []
for info in grid:
    r = tier2.run(info, n_reps=2500)
    cs_b.append(r['cs_bias']); la_b.append(r['la_bias']); cov.append(r['cov'])
    din.append(r['din']); se_est.append(r['se_la']); se_mc.append(r['la_sd'])
cs_b, la_b, cov = map(np.array, (cs_b, la_b, cov))
din, se_est, se_mc = map(np.array, (din, se_est, se_mc))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Tier 2 (full panel micro-sim + estimated Sigma_hat): the spine survives real estimation",
             fontsize=12)

a = ax[0, 0]
a.axhline(0, color="gray", ls=":", lw=.8)
a.plot(grid, cs_b, "o-", color="#c0392b", label="CS (targets ATT)")
a.plot(grid, la_b, "s-", color="#2471a3", label="reweighted LATT (ours)")
a.set_title("Point-estimate bias vs truth (=1)")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("bias"); a.legend(fontsize=8)

a = ax[0, 1]
a.axhline(95, color="green", ls="--", lw=.9, label="95% nominal")
a.plot(grid, cov, "o-", color="#e67e22", label="naive CI (uses Sigma_hat)")
a.set_title("Causal coverage of selected LATT")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("coverage (%)")
a.set_ylim(0, 100); a.legend(fontsize=8)

a = ax[1, 0]
a.plot(grid, se_est, "s-", color="#2471a3", label="estimated SE (conditional)")
a.plot(grid, se_mc, "o-", color="#7d3c98", label="true MC spread (marginal)")
a.set_title("Sigma_hat is calibrated where selection is deterministic;\n"
            "gap = selection-induced variance (motivates Layer 1 / splitting)")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("SE of LATT"); a.legend(fontsize=8)

a = ax[1, 1]
a.plot(grid, din, "^-", color="#555555")
a.set_title("Mechanism: dirty cohorts slipping into selection\n(out of 3 dirty)")
a.set_xlabel("pre-trend informativeness (info)"); a.set_ylabel("avg # dirty selected")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("tier2.png", dpi=130)
print("Saved -> tier2.png")
