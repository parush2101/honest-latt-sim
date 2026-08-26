"""CHC application figure: cohort credibility map + ATT-vs-LATT event study (both weightings)."""
import pandas as pd, numpy as np, json
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_csv("/tmp/chc_slim.csv")
Y={}; POP={}
for r in df.itertuples(): Y[(int(r.fips),int(r.year))]=np.log(r.amr_eld); POP[(int(r.fips),int(r.year))]=r.copop
ft=df.dropna(subset=['chc_year_exp']).drop_duplicates('fips').set_index('fips')['chc_year_exp'].astype(int).to_dict()
never=[f for f in df.fips.unique() if f not in ft]
byg=defaultdict(list)
for f,g in ft.items(): byg[g].append(f)
slopes={int(k):v for k,v in json.load(open("/tmp/chc_slopes.json")).items()}
early=[g for g in byg if 1965<=g<=1974]; credible=[g for g in early if abs(slopes.get(g,9))<0.006]
def nm(y,w): 
    p=[(Y[(f,y)],POP.get((f,y),1) if w else 1) for f in never if (f,y) in Y]
    v=np.array([x[0] for x in p]); wt=np.array([x[1] for x in p]); return np.sum(v*wt)/wt.sum()
def es(cohorts,EV,w):
    out={}
    for e in EV:
        num=den=0.0
        for g in cohorts:
            m=[(Y[(f,g+e)]-Y[(f,g-1)],POP.get((f,g-1),1) if w else 1) for f in byg[g] if (f,g+e) in Y and (f,g-1) in Y]
            if not m: continue
            v=np.array([x[0] for x in m]); wt=np.array([x[1] for x in m])
            dt=np.sum(v*wt)/wt.sum()-(nm(g+e,w)-nm(g-1,w))
            wg=sum(POP.get((f,g-1),1) if w else 1 for f in byg[g]); num+=wg*dt; den+=wg
        out[e]=100*num/den if den else np.nan
    return out
EV=list(range(-6,8))
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
fig.suptitle("Community Health Centers (Bailey & Goodman-Bacon): the credible subpopulation de-attenuates the mortality effect\n"
             "log elderly mortality, differential vs never-treated counties; CHC establishment 1965-74",fontsize=10.5)
# Panel 1: credibility map (per-cohort pre-trend slope)
a=ax[0]
gg=sorted([g for g in early],key=lambda g:abs(slopes.get(g,0)))
cols=["#c0392b" if abs(slopes.get(g,0))>=0.006 else "#2471a3" for g in gg]
a.barh([str(g)+f" (n={len(byg[g])})" for g in gg],[abs(slopes.get(g,0)) for g in gg],color=cols)
a.axvline(0.006,color="green",ls="--",lw=.8,label="credibility threshold")
a.set_xlabel("|differential pre-trend slope|"); a.set_title("Cohort credibility map\n(red = suspect, dropped)"); a.legend(fontsize=7)
# Panel 2: unweighted event study ATT vs LATT
for a2,w,ttl in [(ax[1],False,"Equal-weighted"),(ax[2],True,"Population-weighted")]:
    ba=es(early,EV,w); bc=es(credible,EV,w)
    a2.axhline(0,color="gray",lw=.6,ls=":"); a2.axvline(0,color="gray",lw=.6,ls=":")
    a2.plot(EV,[ba[e] for e in EV],"o-",color="#c0392b",ms=3,label="ATT (all early)")
    a2.plot(EV,[bc[e] for e in EV],"s-",color="#2471a3",ms=3,label="LATT (credible)")
    a2.set_xlabel("years since CHC establishment"); a2.set_ylabel("mortality gap vs never-treated (%)")
    a2.set_title(f"{ttl} event study"); a2.legend(fontsize=7)
plt.tight_layout(rect=[0,0,1,0.90]); plt.savefig("chc_diagnostic.png",dpi=130)
print("saved chc_diagnostic.png")
# print the numbers for the paper
for w,lab in [(False,"equal"),(True,"pop")]:
    ba=es(early,EV,w); bc=es(credible,EV,w)
    pa=np.mean([ba[e] for e in range(1,6)]); pc=np.mean([bc[e] for e in range(1,6)])
    print(f"{lab}-weighted avg post e1-5:  ATT={pa:+.2f}%  LATT={pc:+.2f}%")
