import openpyxl, numpy as np, math
from collections import defaultdict
# recreational cannabis LEGALIZATION effective year by state abbrev
REC={'CO':2012,'WA':2012,'AK':2015,'OR':2015,'CA':2016,'NV':2016,'MA':2016,'ME':2016,
'MI':2018,'VT':2018,'IL':2020,'AZ':2020,'MT':2021,'NJ':2021,'NY':2021,'VA':2021,'NM':2021,'CT':2021,
'RI':2022,'MO':2022,'MD':2023,'DE':2023,'MN':2023,'OH':2023}
emp=defaultdict(dict); stab={}
for yy in range(15,24):
    wb=openpyxl.load_workbook(f'laucnty{yy}.xlsx',read_only=True); ws=wb.active
    for i,row in enumerate(ws.iter_rows(values_only=True)):
        if i<2 or not row[1]: continue
        sf,cf=row[1],row[2]
        if not (isinstance(sf,str) and sf.isdigit()): continue
        fips=sf+cf; 
        try: y=int(row[4]); e=float(row[6])
        except: continue
        if e<=0: continue
        emp[fips][y]=math.log(e)
        nm=row[3] or ''; 
        if ',' in nm: stab[fips]=nm.split(',')[-1].strip()
    wb.close()
YRS=list(range(2015,2024))
# keep counties with full panel
cty=[f for f in emp if all(y in emp[f] for y in YRS)]
onset={}
for f in cty:
    ab=stab.get(f); onset[f]=REC.get(ab,0)
NT=[f for f in cty if onset[f]==0]
print(f"counties full panel: {len(cty)}  never-treated: {len(NT)}")
ntser={y:np.mean([emp[f][y] for f in NT]) for y in YRS}
from collections import Counter
tc=Counter(onset[f] for f in cty if onset[f]>0)
print("cohort county counts:", dict(sorted(tc.items())))
def es(fs,g,e):
    y=g+e; ref=g-1
    if y<2015 or y>2023 or ref<2015: return None
    d=[emp[f][y]-emp[f][ref] for f in fs]
    return np.mean(d)-(ntser[y]-ntser[ref])
cohorts=[g for g in sorted(tc) if 2018<=g<=2022]
byco={g:[f for f in cty if onset[f]==g] for g in cohorts}
print(f"\n{'g':>5} {'n':>4} | maxpre | pre path | post path (avg)")
info={}
for g in cohorts:
    fs=byco[g]
    pre=[es(fs,g,e) for e in range(-3,0)]; pre=[x for x in pre if x is not None]
    post=[es(fs,g,e) for e in range(1,4)]; post=[x for x in post if x is not None]
    mp=max(abs(x) for x in pre) if pre else float('nan')
    pa=np.mean(post) if post else float('nan')
    info[g]=(len(fs),mp,pa)
    print(f"{g:>5} {len(fs):>4} | {mp:.3f} | pre={['%.3f'%x for x in pre]} | post={['%.3f'%x for x in post]} avg={pa:.3f}")
gs=[g for g in cohorts if not math.isnan(info[g][2])]
ns=np.array([info[g][0] for g in gs]); mp=np.array([info[g][1] for g in gs]); pa=np.array([info[g][2] for g in gs])
att=np.average(pa,weights=ns)
print(f"\nPooled ATT (log employment) = {att:.4f}")
for c in [0.01,0.02,0.03,0.05]:
    keep=mp<=c
    if keep.sum()==0: continue
    latt=np.average(pa[keep],weights=ns[keep])
    print(f"  c={c}: keep {keep.sum()}/{len(gs)} LATT={latt:.4f} dropped={[int(gs[i]) for i in range(len(gs)) if not keep[i]]}")
