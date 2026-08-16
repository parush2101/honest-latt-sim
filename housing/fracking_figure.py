import openpyxl, csv, math, numpy as np, random
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
random.seed(7); np.random.seed(7)
# ---- fracking onset (USDA) ----
YRS=list(range(2000,2012))
rows=list(csv.DictReader(open('../honest-latt-sim/shale/usda_oilgas_2000_2011.csv'))) if False else list(csv.DictReader(open('shale/usda_oilgas_2000_2011.csv')))
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
# ---- FHFA county HPI ----
wb=openpyxl.load_workbook('housing/fhfa_county.xlsx',read_only=True); ws=wb.active
hpi=defaultdict(dict)
for i,r in enumerate(ws.iter_rows(values_only=True)):
    if i<7: continue
    fips=r[2]
    if not (isinstance(fips,str) and len(fips)==5 and fips.isdigit()): continue
    try: y=int(r[3]); v=float(r[7])
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
cohorts=[g for g in sorted(set(ons.values())) if 2003<=g<=2011]
byco={g:[f for f in ons if ons[f]==g] for g in cohorts}
EVENTS=list(range(-5,5))
def att_ge(fs,g,e,nt):
    y=g+e; ref=g-1
    if y<1998 or y>2019: return None
    d=[lv[f][y]-lv[f][ref] for f in fs if y in lv[f]]
    return np.mean(d)-(nt[y]-nt[ref])
def ntser(sampleNT):
    return {y:np.mean([lv[f][y] for f in sampleNT]) for y in PY}
def eventpath(sm,nt,gs):
    out={}
    for e in EVENTS:
        vals=[]; ws=[]
        for g in gs:
            a=att_ge(sm[g],g,e,nt)
            if a is not None: vals.append(a); ws.append(len(sm[g]))
        out[e]=np.average(vals,weights=ws) if vals else np.nan
    return out
def maxpre(fs,g,nt):
    v=[abs(att_ge(fs,g,e,nt)) for e in range(-5,-1)]; v=[x for x in v if x is not None]
    return max(v)
nt0=ntser(NT)
mps={g:maxpre(byco[g],g,nt0) for g in cohorts}
C=0.03
CRED=[g for g in cohorts if mps[g]<=C]
def eff(path): 
    v=[path[e] for e in (1,2,3,4) if not np.isnan(path[e])]; return np.mean(v)
pooled=eventpath(byco,nt0,cohorts); cred=eventpath(byco,nt0,CRED)
att0=eff(pooled); latt0=eff(cred)
# credibility path: LATT vs c
cgrid=np.linspace(0.01,0.12,24); lattpath=[]
for c in cgrid:
    keep=[g for g in cohorts if mps[g]<=c]
    lattpath.append(eff(eventpath(byco,nt0,keep)) if keep else np.nan)
# bootstrap
B=500
poolB=defaultdict(list); credB=defaultdict(list); attB=[]; lattB=[]; divB=[]; pathB=[]
for b in range(B):
    sm={g:[random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb=[random.choice(NT) for _ in NT]; nt=ntser(ntb)
    pp=eventpath(sm,nt,cohorts); cc=eventpath(sm,nt,CRED)
    for e in EVENTS: poolB[e].append(pp[e]); credB[e].append(cc[e])
    a=eff(pp); l=eff(cc); attB.append(a); lattB.append(l); divB.append(a-l)
    row=[]
    for c in cgrid:
        keep=[g for g in cohorts if mps[g]<=c]
        row.append(eff(eventpath(sm,nt,keep)) if keep else np.nan)
    pathB.append(row)
def ci(arr): a=np.array(arr); return np.nanpercentile(a,2.5),np.nanpercentile(a,97.5),np.nanstd(a)
attse=np.std(attB); lattse=np.std(lattB); divse=np.std(divB)
pathB=np.array(pathB); pathlo=np.nanpercentile(pathB,2.5,axis=0); pathhi=np.nanpercentile(pathB,97.5,axis=0)
print(f"ATT={att0:.4f} se={attse:.4f} t={att0/attse:.2f}")
print(f"LATT(c={C}, cohorts {CRED}, n={sum(len(byco[g]) for g in CRED)})={latt0:.4f} se={lattse:.4f} t={latt0/lattse:.2f}")
print(f"Divergence={att0-latt0:.4f} se={divse:.4f} t={(att0-latt0)/divse:.2f}")
print("maxpre:",{g:round(mps[g],3) for g in cohorts})

# ---------- FIGURE ----------
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
# Panel 1: credibility map
gg=list(cohorts); mv=[mps[g] for g in gg]
cols=['#2c7fb8' if mps[g]<=C else '#d95f0e' for g in gg]
ax[0].bar(range(len(gg)),mv,color=cols,edgecolor='white')
ax[0].axhline(C,ls='--',color='0.35',lw=1)
ax[0].text(len(gg)-0.4,C+0.002,f'screen $c={C}$',ha='right',va='bottom',fontsize=8.5,color='0.3')
ax[0].set_xticks(range(len(gg))); ax[0].set_xticklabels(gg,rotation=45,fontsize=8)
ax[0].set_ylabel('max pre-trend $|\\hat\\beta_{g,pre}|$'); ax[0].set_title('(a) Cohort credibility',fontsize=10.5)
ax[0].text(0.02,0.95,'retained',color='#2c7fb8',transform=ax[0].transAxes,fontsize=8.5,va='top')
ax[0].text(0.02,0.87,'dropped',color='#d95f0e',transform=ax[0].transAxes,fontsize=8.5,va='top')
# Panel 2: event study
ev=EVENTS
pl=[pooled[e] for e in ev]; cl=[cred[e] for e in ev]
plo=[np.nanpercentile(poolB[e],2.5) for e in ev]; phi=[np.nanpercentile(poolB[e],97.5) for e in ev]
clo=[np.nanpercentile(credB[e],2.5) for e in ev]; chi=[np.nanpercentile(credB[e],97.5) for e in ev]
ax[1].axhline(0,color='0.8',lw=.8); ax[1].axvline(-0.5,color='0.8',lw=.8,ls=':')
ax[1].fill_between(ev,plo,phi,color='#d95f0e',alpha=.15)
ax[1].fill_between(ev,clo,chi,color='#2c7fb8',alpha=.15)
ax[1].plot(ev,pl,'-o',color='#d95f0e',ms=3.5,label='all cohorts (ATT)')
ax[1].plot(ev,cl,'-s',color='#2c7fb8',ms=3.5,label='credible (LATT)')
ax[1].set_xlabel('event time'); ax[1].set_ylabel('log house price index'); ax[1].set_title('(b) Event study',fontsize=10.5)
ax[1].legend(frameon=False,fontsize=8.5,loc='upper left')
# Panel 3: credibility path
ax[2].axhline(att0,color='#d95f0e',ls='--',lw=1)
ax[2].text(cgrid[-1],att0,'  pooled ATT',color='#d95f0e',fontsize=8.5,va='center')
ax[2].axhline(0,color='0.8',lw=.8)
ax[2].fill_between(cgrid,pathlo,pathhi,color='#2c7fb8',alpha=.15)
ax[2].plot(cgrid,lattpath,'-',color='#2c7fb8',lw=1.6)
ax[2].set_xlabel('screen threshold $c$'); ax[2].set_ylabel('credible LATT'); ax[2].set_title('(c) Credibility path',fontsize=10.5)
plt.tight_layout(); plt.savefig('fracking_diagnostic.png',dpi=150,bbox_inches='tight')
print("saved fracking_diagnostic.png")
