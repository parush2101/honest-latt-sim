import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Armstrong-Kolesar level-bound FLCI half-width: h = s * cv_alpha(B/s),
# where cv_alpha(t) = (1-alpha) quantile of |N(t,1)|.
def cv(t, alpha=0.05):
    if t < 1e-9:
        return norm.ppf(1 - alpha/2)
    f = lambda x: (norm.cdf(x - t) - norm.cdf(-x - t)) - (1 - alpha)
    return brentq(f, 0, t + 20)

z = norm.ppf(0.975)
print(f"cv(0) = z_.975 = {cv(0):.4f}  (sanity, want 1.9600)")

# --- Analytic crossing: clean-regime LATT vs honest ATT, gamma=0, ML=0 ---
# h_L = sigma * cv(ML/sigma) = sigma * z         (select {clean}: unbiased, SE sigma)
# h_A = (sigma/sqrt2) * cv( V/(sqrt2 sigma) )     (both cohorts: bias V/2, SE sigma/sqrt2)
# crossing: z = (1/sqrt2) cv(V/(sqrt2 sigma))  ->  cv(t*) = sqrt2 * z,  V* = sqrt2 sigma t*
tstar = brentq(lambda t: cv(t) - np.sqrt(2)*z, 0, 20)
print(f"t* solving cv(t*)=sqrt2*z: {tstar:.4f}")
print(f"INTERVAL threshold: V* = sqrt2*sigma*t* = {np.sqrt(2)*tstar:.4f} * sigma")
print(f"MSE threshold (gamma=0):        V  = sqrt2*sigma   = {np.sqrt(2):.4f} * sigma")

# --- Monte Carlo check in the clean regime (screen drops confounded w.p.~1) ---
rng = np.random.default_rng(1)
def mc(V, sigma, ML=0.0, N=400000, alpha=0.05):
    truth = 1.0
    # LATT selects {clean}: estimate ~ N(truth, sigma^2); honest bound ML
    latt = rng.normal(truth, sigma, N)
    hL = sigma * cv(ML/sigma, alpha)
    covL = np.mean(np.abs(latt - truth) <= hL)
    # ATT uses both cohorts: clean N(truth,sig^2) + confounded N(truth+V,sig^2), avg
    b1 = rng.normal(truth, sigma, N); b2 = rng.normal(truth + V, sigma, N)
    att = 0.5*(b1+b2)                       # ~ N(truth + V/2, sigma^2/2)
    sA = sigma/np.sqrt(2); BA = V/2
    hA = sA * cv(BA/sA, alpha)
    covA = np.mean(np.abs(att - truth) <= hA)
    return covL, 2*hL, covA, 2*hA

print("\n sigma=1; sweep V.  (both should cover ~0.95; compare full widths)")
print(f"{'V':>5} | {'covL':>6} {'widL':>7} | {'covA':>6} {'widA':>7} | shorter")
for V in [0.5,1.0,1.5,1.63,2.0,2.5,3.0]:
    cL,wL,cA,wA = mc(V, 1.0)
    print(f"{V:5.2f} | {cL:6.3f} {wL:7.3f} | {cA:6.3f} {wA:7.3f} | {'LATT' if wL<wA else 'ATT'}")

print("\n effect of an honest residual buffer ML on the LATT interval (sigma=1):")
for ML in [0.0, 0.1, 0.2]:
    Vstar = brentq(lambda V: 1.0*cv(ML/1.0) - (1/np.sqrt(2))*cv(V/np.sqrt(2)), 1e-6, 20)
    print(f"   ML={ML:.2f}*sigma -> interval threshold V* = {Vstar:.3f}*sigma")
