import csv, random, numpy as np
from collections import defaultdict
rows=list(csv.DictReader(open('shale_panel.csv')))
lpci=defaultdict(dict); onset={}
for r in rows:
    f=r['fips']; lpci[f][int(r['year'])]=float(r['lpci']); onset[f]=int(r['onset'])
NT=[f for f in onset if onset[f]==0]
def es_cohort(fs,g,ntser):
    ref=g-1; out=[]
    for e in range(1,5):
        y=g+e
        if y>2019: continue
        d=[lpci[f][y]-lpci[f][ref] for f in fs if y in lpci[f] and ref in lpci[f]]
        out.append(np.mean(d)-(ntser[y]-ntser[ref]))
    return np.mean(out)
cohorts=[g for g in sorted(set(onset.values())) if 2003<=g<=2011]
byco={g:[f for f in onset if onset[f]==g] for g in cohorts}
KEEP=[2003,2006,2008]
def agg(sm,ntser,gs):
    ns=[len(sm[g]) for g in gs]; ef=[es_cohort(sm[g],g,ntser) for g in gs]
    return np.average(ef,weights=ns)
ntser0={y:np.mean([lpci[f][y] for f in NT if y in lpci[f]]) for y in range(1998,2020)}
att0=agg(byco,ntser0,cohorts); latt0=agg(byco,ntser0,KEEP)
B=600; atts=[];latts=[]; random.seed(3)
for b in range(B):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NT) for _ in NT]
    ns={y:np.mean([lpci[f][y] for f in ntb if y in lpci[f]]) for y in range(1998,2020)}
    atts.append(agg(sm,ns,cohorts)); latts.append(agg(sm,ns,KEEP))
atts=np.array(atts); latts=np.array(latts)
for nm,pt,bs in [("ATT(all 9)",att0,atts),("LATT{2003,06,08}",latt0,latts)]:
    se=bs.std(); print(f"{nm:20s} est={pt:.4f} se={se:.4f} t={pt/se:.2f} CI=[{pt-1.96*se:.4f},{pt+1.96*se:.4f}] M*={abs(pt)-1.96*se:.4f}")
print("\nDivergence ATT-LATT =",round(att0-latt0,4))
