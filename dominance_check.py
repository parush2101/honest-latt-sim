import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)

def p2(phi, V, c, sigma):
    # prob confounded cohort passes flatness screen |beta_pre| <= c
    return norm.cdf((c - phi*V)/sigma) - norm.cdf((-c - phi*V)/sigma)

def mse_montecarlo(phi, V, gamma, c, sigma, N=4_000_000):
    # target = ATT = tau_bar (set = 1). tau1 = 1 - gamma/2, tau2 = 1 + gamma/2
    tau1, tau2 = 1 - gamma/2, 1 + gamma/2
    # clean cohort (g=1): delta_pre=0, delta_post=0
    b1_pre  = rng.normal(0.0,        sigma, N)
    b1_post = rng.normal(tau1,       sigma, N)
    # confounded cohort (g=2): delta_pre=phi*V, delta_post=V
    b2_pre  = rng.normal(phi*V,      sigma, N)
    b2_post = rng.normal(tau2 + V,   sigma, N)
    keep1 = np.abs(b1_pre) <= c
    keep2 = np.abs(b2_pre) <= c
    # ATT estimator: always both
    att = 0.5*(b1_post + b2_post)
    # LATT estimator: average over kept; if none kept, fall back to all
    num = keep1*b1_post + keep2*b2_post
    den = keep1.astype(float) + keep2.astype(float)
    latt = np.where(den>0, num/np.maximum(den,1), att)
    mse_att  = np.mean((att  - 1.0)**2)
    mse_latt = np.mean((latt - 1.0)**2)
    return mse_att, mse_latt

def delta_formula(phi, V, gamma, c, sigma):
    return (1 - p2(phi,V,c,sigma))/4.0 * (V**2 - gamma**2 - 2*sigma**2)

print(f"{'phi':>5} {'V':>5} {'gam':>5} {'sig':>5} | {'p2':>6} | {'MC Delta':>10} {'formula':>10} {'sign':>6}")
for (phi,V,gamma,c,sigma) in [
    (0.0, 0.6, 0.0, 0.4, 0.20),
    (0.5, 0.6, 0.0, 0.4, 0.20),
    (1.0, 0.6, 0.0, 0.4, 0.20),
    (1.5, 0.6, 0.0, 0.4, 0.20),
    (1.5, 0.30,0.0, 0.4, 0.20),   # small V: V^2=0.09 < 2sig^2=0.08? no 0.09>0.08 barely
    (1.5, 0.25,0.0, 0.4, 0.20),   # V^2=0.0625 < 2sig^2=0.08 -> LATT should be WORSE
    (1.5, 0.6, 0.5, 0.4, 0.20),   # heterogeneity gamma=0.5: V^2=.36 vs gam^2+2sig^2=.25+.08=.33 -> barely dominates
    (1.5, 0.6, 0.7, 0.4, 0.20),   # gamma=0.7: .36 vs .49+.08=.57 -> LATT worse
]:
    mc_att, mc_latt = mse_montecarlo(phi,V,gamma,c,sigma)
    mc_delta = mc_att - mc_latt
    f = delta_formula(phi,V,gamma,c,sigma)
    print(f"{phi:5.2f} {V:5.2f} {gamma:5.2f} {sigma:5.2f} | {p2(phi,V,c,sigma):6.3f} | {mc_delta:10.5f} {f:10.5f} {'LATT+' if mc_delta>0 else 'ATT+':>6}")
