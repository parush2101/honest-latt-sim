"""Robust re-run: report %finite + median selective length; add 1-cohort sanity check."""
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr
rng = np.random.default_rng(7)
z = norm.ppf(0.975)

def _phidiff(lo, hi):
    if hi <= lo: return 0.0
    if hi < 0:   return norm.cdf(hi) - norm.cdf(lo)
    if lo > 0:   return norm.sf(lo) - norm.sf(hi)
    return norm.cdf(hi) - norm.cdf(lo)

def selective_interval(bhat, Sigma, eta, constraints, alpha=0.05):
    T = float(eta @ bhat); s2 = float(eta @ Sigma @ eta)
    if s2 <= 0: return T, T, 0.0
    sig = np.sqrt(s2); c_dir = Sigma @ eta / s2; Z = bhat - c_dir * T
    Vlo, Vhi = -np.inf, np.inf
    for a, b in constraints:
        aj = float(a @ c_dir); resid = b - float(a @ Z)
        if aj > 1e-12:   Vhi = min(Vhi, resid/aj)
        elif aj < -1e-12: Vlo = max(Vlo, resid/aj)
    Vlo = min(Vlo, T); Vhi = max(Vhi, T)
    def pivot(th):
        a=(Vlo-th)/sig; b=(Vhi-th)/sig; t=(T-th)/sig
        den=_phidiff(a,b)
        if den<=1e-290:
            return 1.0 if th < T else 0.0   # degenerate: decide by side
        return _phidiff(a,t)/den
    def solve(level):
        lo,hi = T-15*sig, T+15*sig
        for _ in range(80):
            mid=0.5*(lo+hi); fm=pivot(mid)
            if fm>level: lo=mid
            else: hi=mid
        return 0.5*(lo+hi)
    hi_end=solve(alpha/2); lo_end=solve(1-alpha/2)
    lo_end,hi_end=min(lo_end,hi_end),max(lo_end,hi_end)
    L=hi_end-lo_end
    finite = L < 12*sig            # flag boundary blow-up
    return lo_end, hi_end, L, finite

def build_constraints(bpre, sel, c, G):
    cons=[]
    for g in range(G):
        e=np.zeros(2*G); e[g]=1.0
        if sel[g]:
            cons.append((e.copy(),c)); cons.append((-e.copy(),c))
        else:
            cons.append((-e.copy(),-c) if bpre[g]>c else (e.copy(),-c))
    return cons

def run(s, delta_pre, c, rho, n_reps):
    G=len(delta_pre); theta=np.ones(G)
    beta=np.concatenate([delta_pre.astype(float), theta])
    Sigma=np.zeros((2*G,2*G))
    for g in range(G):
        Sigma[g,g]=s**2; Sigma[G+g,G+g]=s**2; Sigma[g,G+g]=rho*s**2; Sigma[G+g,g]=rho*s**2
    Lc=np.linalg.cholesky(Sigma)
    cn=cs=cl=0; ln=ls=0.0; sel_lens=[]; nfin=0
    for _ in range(n_reps):
        bhat=beta+Lc@rng.standard_normal(2*G); bpre=bhat[:G]
        sel=np.abs(bpre)<=c
        if sel.sum()==0: sel=np.ones(G,bool)
        tgt=1.0
        m=bhat[G:][sel].mean(); se=s/np.sqrt(sel.sum())
        cn+=(m-z*se<=tgt<=m+z*se); ln+=2*z*se
        d1=Lc@rng.standard_normal(2*G)*np.sqrt(2); d2=Lc@rng.standard_normal(2*G)*np.sqrt(2)
        s1=np.abs((beta+d1)[:G])<=c
        if s1.sum()==0: s1=np.ones(G,bool)
        m2=(beta+d2)[G:][s1].mean(); se2=(s*np.sqrt(2))/np.sqrt(s1.sum())
        cs+=(m2-z*se2<=1.0<=m2+z*se2); ls+=2*z*se2
        eta=np.zeros(2*G); eta[G+np.where(sel)[0]]=1.0/sel.sum()
        lo,hi,L,fin=selective_interval(bhat,Sigma,eta,build_constraints(bpre,sel,c,G))
        cl+=(lo<=tgt<=hi); nfin+=fin
        if fin: sel_lens.append(L)
    N=n_reps
    return dict(cn=100*cn/N,cs=100*cs/N,cl=100*cl/N,ln=ln/N,ls=ls/N,
                sel_med=np.median(sel_lens) if sel_lens else np.inf,
                pfin=100*nfin/N)

# 1-cohort sanity check (RR Example 4 regime): selection = |bpre|<=c on a single cohort
print("=== 1-cohort sanity check (selective should be finite & ~95%) ===")
r=run(0.25, np.array([0.30]), 0.40, 0.5, 4000)
print(f"  naive cov {r['cn']:.1f}  split {r['cs']:.1f}  selective {r['cl']:.1f} | "
      f"%finite {r['pfin']:.0f}  median sel len {r['sel_med']:.3f}  split len {r['ls']:.3f}")

print("\n=== 8-cohort aggregate race (median length among finite; %finite) ===")
G=8; dp=np.linspace(0,0.60,G)
print(f"{'s':>5} | {'naive':>6}{'split':>7}{'selctv':>7} | {'%fin':>5} {'medSel':>7}{'split':>7}{'naive':>7}")
for s in [0.30,0.20,0.12]:
    r=run(s,dp,0.40,0.5,4000)
    print(f"{s:5.2f} | {r['cn']:6.1f}{r['cs']:7.1f}{r['cl']:7.1f} | "
          f"{r['pfin']:5.0f} {r['sel_med']:7.3f}{r['ls']:7.3f}{r['ln']:7.3f}")
