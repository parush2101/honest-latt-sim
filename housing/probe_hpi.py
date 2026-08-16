import openpyxl, csv, math, numpy as np, random
from collections import defaultdict
# fracking onset from USDA (reuse shale logic)
YRS=list(range(2000,2012))
rows=list(csv.DictReader(open('../shale/usda_oilgas_2000_2011.csv')))
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
# FHFA county HPI (log HPI col F index base-100-at-first)
wb=openpyxl.load_workbook('fhfa_county.xlsx',read_only=True); ws=wb.active
hpi=defaultdict(dict)
for i,r in enumerate(ws.iter_rows(values_only=True)):
    if i<7: continue
    fips=r[2]; 
    if not (isinstance(fips,str) and len(fips)==5 and fips.isdigit()): continue
    try: y=int(r[3]); v=float(r[7])  # HPI 2000 base (col H) for comparability
    except: continue
    if v>0: hpi[fips][y]=math.log(v)
PY=list(range(1998,2020))
lv=defaultdict(dict); ons={}
for f in hpi:
    if f in decline: continue
    if any(y not in hpi[f] for y in PY): continue
    for y in PY: lv[f][y]=hpi[f][y]
    ons[f]=onset.get(f,0)
NT=[f for f in ons if ons[f]==0]
print(f"counties w/ full HPI panel: {len(ons)}  boom retained: {sum(1 for f in ons if ons[f]>0)}  NT: {len(NT)}")
def es_c(fs,g,nt):
    ref=g-1; out=[]
    for e in range(1,5):
        y=g+e
        if y>2019: continue
        d=[lv[f][y]-lv[f][ref] for f in fs if y in lv[f] and ref in lv[f]]
        if d: out.append(np.mean(d)-(nt[y]-nt[ref]))
    return np.mean(out) if out else float('nan')
def mpre(fs,g,nt):
    ref=g-1;v=[]
    for e in range(-5,-1):
        y=g+e
        if y<1998: continue
        d=[lv[f][y]-lv[f][ref] for f in fs if y in lv[f] and ref in lv[f]]
        if d: v.append(abs(np.mean(d)-(nt[y]-nt[ref])))
    return max(v) if v else float('nan')
cohorts=[g for g in sorted(set(ons.values())) if 2003<=g<=2011]
byco={g:[f for f in ons if ons[f]==g] for g in cohorts}
nt0={y:np.mean([lv[f][y] for f in NT if y in lv[f]]) for y in PY}
print(f"\n{'g':>5} {'n':>4} maxpre  post-eff")
mps={}; ef={}
for g in cohorts:
    mps[g]=mpre(byco[g],g,nt0); ef[g]=es_c(byco[g],g,nt0)
    print(f"{g:>5} {len(byco[g]):>4} {mps[g]:.3f}  {ef[g]:.3f}")
ns=[len(byco[g]) for g in cohorts]
att=np.average([ef[g] for g in cohorts],weights=ns)
print(f"\nPooled ATT (log HPI) = {att:.4f}")
for c in [0.03,0.05,0.08,0.10]:
    keep=[g for g in cohorts if mps[g]<=c]
    if not keep: continue
    latt=np.average([ef[g] for g in keep],weights=[len(byco[g]) for g in keep])
    print(f"  c={c}: keep {len(keep)} LATT={latt:.4f} dropped={[g for g in cohorts if mps[g]>c]}")

# ---- bootstrap ATT vs LATT(c=0.03 fixed set) and divergence ----
KEEP=[g for g in cohorts if mps[g]<=0.03]  # {2003,2004,2005}
print(f"\nFixed credible set (c=0.03): {KEEP}")
def agg(sm,nt,gs): 
    w=[len(sm[g]) for g in gs]; e=[es_c(sm[g],g,nt) for g in gs]
    return np.average(e,weights=w)
att0=agg(byco,nt0,cohorts); latt0=agg(byco,nt0,KEEP)
B=600; A=[];L=[];D=[]; random.seed(11)
for b in range(B):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NT) for _ in NT]
    nt={y:np.mean([lv[f][y] for f in ntb if y in lv[f]]) for y in PY}
    a=agg(sm,nt,cohorts); l=agg(sm,nt,KEEP)
    A.append(a);L.append(l);D.append(a-l)
A=np.array(A);L=np.array(L);D=np.array(D)
print(f"ATT  = {att0:.4f}  se={A.std():.4f}  t={att0/A.std():.2f}")
print(f"LATT = {latt0:.4f}  se={L.std():.4f}  t={latt0/L.std():.2f}  M*={abs(latt0)-1.96*L.std():.4f}")
print(f"Divergence ATT-LATT = {att0-latt0:.4f}  se={D.std():.4f}  t={(att0-latt0)/D.std():.2f}  frac>0={(D>0).mean():.2f}")
