import csv, random, numpy as np
from collections import defaultdict
rows=list(csv.DictReader(open('shale_panel.csv')))
lpci=defaultdict(dict); onset={}
for r in rows:
    f=r['fips']; lpci[f][int(r['year'])]=float(r['lpci']); onset[f]=int(r['onset'])
NT=[f for f in onset if onset[f]==0]
ntser={y:np.mean([lpci[f][y] for f in NT if y in lpci[f]]) for y in range(1998,2020)}
def cohort_es(fs,g):
    ref=g-1; out={}
    for e in range(1,5):
        y=g+e
        if y>2019: continue
        mt=np.mean([lpci[f][y]-lpci[f][ref] for f in fs if y in lpci[f] and ref in lpci[f]])
        out[e]=mt-(ntser[y]-ntser[ref])
    return np.mean(list(out.values()))
def maxpre(fs,g):
    ref=g-1; vals=[]
    for e in range(-5,-1):
        y=g+e
        if y<1998: continue
        mt=np.mean([lpci[f][y]-lpci[f][ref] for f in fs if y in lpci[f] and ref in lpci[f]])
        vals.append(abs(mt-(ntser[y]-ntser[ref])))
    return max(vals)
cohorts=[g for g in sorted(set(onset.values())) if 2003<=g<=2011]
byco={g:[f for f in onset if onset[f]==g] for g in cohorts}
def estimate(sample_map, c=0.03):
    # sample_map: cohort->list of fips (possibly resampled)
    # recompute NT series from resampled NT too
    global ntser
    # ATT and LATT
    effs={g:cohort_es(sample_map[g],g) for g in cohorts}
    mps={g:maxpre(sample_map[g],g) for g in cohorts}
    ns={g:len(sample_map[g]) for g in cohorts}
    att=np.average([effs[g] for g in cohorts],weights=[ns[g] for g in cohorts])
    keep=[g for g in cohorts if mps[g]<=c]
    latt=np.average([effs[g] for g in keep],weights=[ns[g] for g in keep]) if keep else np.nan
    return att,latt,keep
att0,latt0,keep0=estimate(byco)
print(f"Point: ATT={att0:.4f}  LATT(c=.03)={latt0:.4f}  kept={keep0}")
# cluster bootstrap over counties (resample within each cohort and NT)
B=400; atts=[]; latts=[]; random.seed(1)
NTfull=NT
for b in range(B):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NTfull) for _ in NTfull]
    ntser={y:np.mean([lpci[f][y] for f in ntb if y in lpci[f]]) for y in range(1998,2020)}
    a,l,_=estimate(sm)
    atts.append(a); latts.append(l)
ntser={y:np.mean([lpci[f][y] for f in NT if y in lpci[f]]) for y in range(1998,2020)}
atts=np.array(atts); latts=np.array(latts)
print(f"ATT  se={atts.std():.4f}  t={att0/atts.std():.2f}")
print(f"LATT se={latts.std():.4f}  t={latt0/latts.std():.2f}")
se=latts.std(); z=1.96
Mstar=abs(latt0)-z*se
print(f"LATT 95% CI: [{latt0-z*se:.4f}, {latt0+z*se:.4f}]")
print(f"Level-bound breakdown M* (approx) = |LATT|-1.96*se = {Mstar:.4f} log points")
# compare to typical retained-cohort pre-trend magnitude
mps={g:maxpre(byco[g],g) for g in keep0}
print(f"Retained cohorts' max pre-trend: {mps}  (M* should exceed these to be 'robust')")
