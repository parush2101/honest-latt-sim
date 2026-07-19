"""
Sharper distortion test: (1) measure selection distortion directly on the FLCI CENTER
(the estimator v'beta_hat_agg), which is not masked by the conservative width; and
(2) evaluate coverage with M tuned near the residual curvature so the FLCI sits at the
~95% boundary where distortion actually bites. Strong pre/post correlation (rho=0.7) to
maximize any distortion. full-data vs split.
"""
import numpy as np
from scipy.stats import norm
import layer2_full as L2

TARGET_E = 3
l_post = np.zeros(L2.npost); l_post[TARGET_E-1] = 1.0
L2.V_VEC, L2.W_PRE = L2.extrapolation_weights(l_post)
bbar1 = L2.max_bias_SD(L2.V_VEC, 1.0)
npre, npost = L2.npre, L2.npost
z = norm.ppf(0.975); pre_e, post_e = L2.pre_e, L2.post_e

def cohort_delta(C): return np.array([0.5*C*(e**2) for e in np.concatenate([pre_e, post_e])])
def within_cov(sigma, rho, n):
    idx=np.arange(n); R=rho**np.abs(idx[:,None]-idx[None,:]); return (sigma**2)*R

def run(C_marg, C_dirty, sigma, rho, c_sel, M_list, n_reps, seed=1):
    rng=np.random.default_rng(seed); G=12
    curv=np.array([0]*4+[C_marg]*4+[C_dirty]*4, float)
    tau=np.concatenate([np.zeros(npre), np.ones(npost)])
    means=np.array([tau+cohort_delta(c) for c in curv])
    Sc=within_cov(sigma,rho,npre+npost); Lc=np.linalg.cholesky(Sc)

    ctr_full=[]; ctr_split=[]; nsel=[]; residC=[]
    cov_full={M:0 for M in M_list}; cov_split={M:0 for M in M_list}
    for _ in range(n_reps):
        B=means+(Lc@rng.standard_normal((npre+npost,G))).T
        sel=np.max(np.abs(B[:,:npre]),axis=1)<=c_sel
        if sel.sum()==0: sel=np.ones(G,bool)
        agg=B[sel].mean(0); Sig=Sc/sel.sum()
        c=float(L2.V_VEC@agg); sv=np.sqrt(L2.V_VEC@Sig@L2.V_VEC)
        ctr_full.append(c); nsel.append(sel.sum()); residC.append(curv[sel].mean())
        for M in M_list:
            h=L2.cv((M*bbar1)/sv)*sv; cov_full[M]+=(c-h<=1.0<=c+h)
        # split
        B1=means+(Lc@rng.standard_normal((npre+npost,G))).T*np.sqrt(2)
        B2=means+(Lc@rng.standard_normal((npre+npost,G))).T*np.sqrt(2)
        s2=np.max(np.abs(B1[:,:npre]),axis=1)<=c_sel
        if s2.sum()==0: s2=np.ones(G,bool)
        agg2=B2[s2].mean(0); Sig2=2*Sc/s2.sum()
        c2=float(L2.V_VEC@agg2); sv2=np.sqrt(L2.V_VEC@Sig2@L2.V_VEC)
        ctr_split.append(c2)
        for M in M_list:
            h=L2.cv((M*bbar1)/sv2)*sv2; cov_split[M]+=(c2-h<=1.0<=c2+h)
    N=n_reps
    return (np.mean(ctr_full)-1, np.mean(ctr_split)-1, np.mean(nsel), np.mean(residC),
            {M:100*cov_full[M]/N for M in M_list}, {M:100*cov_split[M]/N for M in M_list})

if __name__=="__main__":
    M_list=[0.03,0.05,0.08,0.12]
    print("Selection distortion on the FLCI center, and coverage near the 95% boundary")
    print("rho=0.7 (strong pre/post corr, to maximize distortion), tau_3 target\n")
    for C_marg,c_sel in [(0.10,0.80),(0.10,0.70)]:
        bf,bs,ns,rc,cf,csp=run(C_marg,0.35,0.10,0.7,c_sel,M_list,8000)
        print(f"C_marg={C_marg}, c_sel={c_sel}: avg#sel={ns:.2f}, residual curvature={rc:.3f}")
        print(f"   CENTER bias  full={bf:+.4f}   split={bs:+.4f}   (distortion={bf-bs:+.4f})")
        for M in M_list:
            print(f"   M={M:.2f}: coverage full={cf[M]:5.1f}%  split={csp[M]:5.1f}%  "
                  f"gap={csp[M]-cf[M]:+.1f}pp")
        print()
