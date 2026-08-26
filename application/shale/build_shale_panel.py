import csv, glob, os, json, math
YRS=list(range(2000,2012))
# --- onset from USDA ---
rows=list(csv.DictReader(open('usda_oilgas_2000_2011.csv')))
def ser(r,p): return [float(r[f'{p}{y}'] or 0) for y in YRS]
onset={}; boomset=set(); declineset=set(); producing=set()
for r in rows:
    fips=r['FIPS'].zfill(5)
    oil=ser(r,'oil'); gas=ser(r,'gas'); boe=[oil[i]+gas[i]/6 for i in range(len(YRS))]
    if max(boe)>0: producing.add(fips)
    g=r['oil_gas_change_group']
    if g=='H_Decline': declineset.add(fips)
    if g!='H_Growth': continue
    peak=max(boe); base=sum(boe[:3])/3; thr=base+0.25*(peak-base)
    o=None
    for i,y in enumerate(YRS):
        if boe[i]>=thr and boe[i]>base: o=y; break
    if o is not None:
        onset[fips]=o; boomset.add(fips)
# --- BEA per-capita income (LineCode 3) and total (LineCode 1) ---
pci={}; # fips -> {year: value}
tot={}
for f in glob.glob('../walmart/CAINC1_*.csv'):
    for r in csv.reader(open(f, encoding='latin-1')):
        if len(r)<10: continue
        geo=r[0].strip().strip('"')
        if not geo.isdigit() or len(geo)!=5 or geo.endswith('000'): continue
        lc=r[4].strip().strip('"')
        if lc not in ('1','3'): continue
        # header row: columns 8.. correspond to years 1969.. ; find year index
        # year 1969 at col idx 8
        d={}
        for j,val in enumerate(r[8:]):
            y=1969+j
            v=val.strip().strip('"')
            try: d[y]=float(v)
            except: pass
        if lc=='3': pci[geo]=d
        else: tot[geo]=d
print("counties with pci:",len(pci),"boom:",len(boomset))
# --- build panel 1998-2019, per-capita income ---
PANEL_YRS=list(range(1998,2020))
out=[]
allf=set(pci.keys())
# never-treated: not boom, not decline, ideally non-producing OR status-quo producers. Use: not boom & not decline.
for fips in sorted(allf):
    if fips in declineset: continue  # drop contaminated decliners
    o=onset.get(fips,0)
    d=pci.get(fips,{})
    if any(y not in d or d[y]<=0 for y in PANEL_YRS): continue
    for y in PANEL_YRS:
        out.append((fips,y,math.log(d[y]),o,1 if o>0 else 0))
with open('shale_panel.csv','w',newline='') as fo:
    w=csv.writer(fo); w.writerow(['fips','year','lpci','onset','ever_treated'])
    w.writerows(out)
from collections import Counter
treated_fips={r[0] for r in out if r[3]>0}
nt_fips={r[0] for r in out if r[3]==0}
print("panel rows:",len(out),"treated counties:",len(treated_fips),"never-treated:",len(nt_fips))
print("cohort sizes in panel:",sorted(Counter(onset[f] for f in treated_fips).items()))
