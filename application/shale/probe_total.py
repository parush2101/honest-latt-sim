import csv, glob, math, random, numpy as np
from collections import defaultdict
YRS=list(range(2000,2012))
rows=list(csv.DictReader(open('usda_oilgas_2000_2011.csv')))
def ser(r,p): return [float(r[f'{p}{y}'] or 0) for y in YRS]
onset={}; decline=set()
for r in rows:
    f=r['FIPS'].zfill(5)
    if r['oil_gas_change_group']=='H_Decline': decline.add(f)
    if r['oil_gas_change_group']!='H_Growth': continue
    oil=ser(r,'oil'); gas=ser(r,'gas'); boe=[oil[i]+gas[i]/6 for i in range(len(YRS))]
    peak=max(boe); base=sum(boe[:3])/3; thr=base+0.25*(peak-base)
    for i,y in enumerate(YRS):
        if boe[i]>=thr and boe[i]>base: onset[f]=y; break
# total personal income LineCode 1
tot=defaultdict(dict)
for f in glob.glob('../walmart/CAINC1_*.csv'):
    for r in csv.reader(open(f,encoding='latin-1')):
        if len(r)<10: continue
        geo=r[0].strip().strip('"')
        if not (geo.isdigit() and len(geo)==5 and not geo.endswith('000')): continue
        if r[4].strip().strip('"')!='1': continue
        for j,val in enumerate(r[8:]):
            try: tot[geo][1969+j]=float(val.strip().strip('"'))
            except: pass
PY=list(range(1998,2020))
lv=defaultdict(dict); ons={}
for f in tot:
    if f in decline: continue
    if any(y not in tot[f] or tot[f][y]<=0 for y in PY): continue
    for y in PY: lv[f][y]=math.log(tot[f][y])
    ons[f]=onset.get(f,0)
NT=[f for f in ons if ons[f]==0]
def es_c(fs,g,nt):
    ref=g-1; out=[]
    for e in range(1,5):
        y=g+e
        if y>2019: continue
        d=[lv[f][y]-lv[f][ref] for f in fs if y in lv[f] and ref in lv[f]]
        out.append(np.mean(d)-(nt[y]-nt[ref]))
    return np.mean(out)
def mpre(fs,g,nt):
    ref=g-1;v=[]
    for e in range(-5,-1):
        y=g+e
        if y<1998: continue
        d=[lv[f][y]-lv[f][ref] for f in fs if y in lv[f] and ref in lv[f]]
        v.append(abs(np.mean(d)-(nt[y]-nt[ref])))
    return max(v)
cohorts=[g for g in sorted(set(ons.values())) if 2003<=g<=2011]
byco={g:[f for f in ons if ons[f]==g] for g in cohorts}
nt0={y:np.mean([lv[f][y] for f in NT if y in lv[f]]) for y in PY}
mps={g:mpre(byco[g],g,nt0) for g in cohorts}
ef={g:es_c(byco[g],g,nt0) for g in cohorts}
print("cohort: n, maxpre, post-eff")
for g in cohorts: print(f"  {g}: n={len(byco[g])} maxpre={mps[g]:.4f} eff={ef[g]:.4f}")
for c in [0.03,0.04,0.05]:
    keep=[g for g in cohorts if mps[g]<=c]
    ns=[len(byco[g]) for g in cohorts]; att=np.average([ef[g] for g in cohorts],weights=ns)
    if keep:
        latt=np.average([ef[g] for g in keep],weights=[len(byco[g]) for g in keep])
        print(f"c={c}: ATT={att:.4f} LATT={latt:.4f} keep={keep}")
# bootstrap fixed set at best c
KEEP=[g for g in cohorts if mps[g]<=0.03]
def agg(sm,nt,gs): 
    return np.average([es_c(sm[g],g,nt) for g in gs],weights=[len(sm[g]) for g in gs])
att0=agg(byco,nt0,cohorts); latt0=agg(byco,nt0,KEEP)
B=500; A=[];L=[];random.seed(5)
for b in range(B):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NT) for _ in NT]; nt={y:np.mean([lv[f][y] for f in ntb if y in lv[f]]) for y in PY}
    A.append(agg(sm,nt,cohorts)); L.append(agg(sm,nt,KEEP))
A=np.array(A);L=np.array(L)
print(f"\nTOTAL income  KEEP={KEEP}")
print(f"ATT ={att0:.4f} se={A.std():.4f} t={att0/A.std():.2f} M*={abs(att0)-1.96*A.std():.4f}")
print(f"LATT={latt0:.4f} se={L.std():.4f} t={latt0/L.std():.2f} M*={abs(latt0)-1.96*L.std():.4f}")

# paired bootstrap of divergence ATT-LATT and screen-threshold sensitivity
random.seed(9); D=[]
for b in range(600):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NT) for _ in NT]; nt={y:np.mean([lv[f][y] for f in ntb if y in lv[f]]) for y in PY}
    D.append(agg(sm,nt,cohorts)-agg(sm,nt,KEEP))
D=np.array(D)
print(f"\nDivergence ATT-LATT = {att0-latt0:.4f}  se={D.std():.4f}  t={(att0-latt0)/D.std():.2f}  (fraction>0: {(D>0).mean():.2f})")
print("Screen-threshold sensitivity of LATT (total income):")
for c in [0.02,0.025,0.03,0.035,0.04,0.05]:
    keep=[g for g in cohorts if mps[g]<=c]
    if not keep: print(f"  c={c}: (empty)"); continue
    l=np.average([ef[g] for g in keep],weights=[len(byco[g]) for g in keep])
    print(f"  c={c}: LATT={l:.4f}  keep={keep}")
