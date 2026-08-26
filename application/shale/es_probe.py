import csv
from collections import defaultdict
rows=list(csv.DictReader(open('shale_panel.csv')))
lpci=defaultdict(dict); onset={}
for r in rows:
    f=r['fips']; y=int(r['year']); lpci[f][y]=float(r['lpci']); onset[f]=int(r['onset'])
NT=[f for f in onset if onset[f]==0]
def mean(fs,y): 
    v=[lpci[f][y] for f in fs if y in lpci[f]]; return sum(v)/len(v) if v else None
# never-treated mean path
ntpath={y:mean(NT,y) for y in range(1998,2020)}
cohorts=sorted(set(o for o in onset.values() if o>=2003 and o<=2011))
print(f"{'g':>5} {'n':>4} | pre e-5..-2 (maxabs)      | post e+1..+4 avg")
import statistics
res={}
for g in cohorts:
    fs=[f for f in onset if onset[f]==g]
    ref=g-1
    def es(e):
        y=g+e
        if y<1998 or y>2019: return None
        mt=mean(fs,y); mr=mean(fs,ref); nt_y=ntpath.get(y); nt_r=ntpath.get(ref)
        if None in (mt,mr,nt_y,nt_r): return None
        return (mt-mr)-(nt_y-nt_r)
    pre=[es(e) for e in range(-5,-1)]; pre=[x for x in pre if x is not None]
    post=[es(e) for e in range(1,5)]; post=[x for x in post if x is not None]
    maxpre=max(abs(x) for x in pre) if pre else float('nan')
    postavg=sum(post)/len(post) if post else float('nan')
    res[g]=(len(fs),maxpre,postavg,pre,post)
    print(f"{g:>5} {len(fs):>4} | maxpre={maxpre:.4f}  pre={['%.3f'%x for x in pre]} | post={['%.3f'%x for x in post]} avg={postavg:.4f}")
# ATT (all cohorts, simple avg of postavg weighted by n) vs LATT (flattest-pre cohorts)
import numpy as np
gs=list(res.keys()); ns=np.array([res[g][0] for g in gs]); mp=np.array([res[g][1] for g in gs]); pa=np.array([res[g][2] for g in gs])
att=np.average(pa,weights=ns)
print(f"\nPooled ATT (n-wtd post avg): {att:.4f}")
for c in [0.02,0.03,0.04,0.05]:
    keep=mp<=c
    if keep.sum()==0: continue
    latt=np.average(pa[keep],weights=ns[keep])
    print(f"  screen c={c}: keep {keep.sum()}/{len(gs)} cohorts, LATT={latt:.4f}  (dropped g={[gs[i] for i in range(len(gs)) if not keep[i]]})")
