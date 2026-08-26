import csv, numpy as np
from collections import defaultdict
# medical marijuana effective YEAR (first-pass coding; refine if promising)
MML={'California':1996,'Alaska':1998,'Oregon':1998,'Washington':1998,'Maine':1999,
'Colorado':2000,'Hawaii':2000,'Nevada':2000,'Montana':2004,'Vermont':2004,'Rhode Island':2006,
'New Mexico':2007,'Michigan':2008,'New Jersey':2010,'Arizona':2010,'Delaware':2011,
'Connecticut':2012,'Massachusetts':2012,'New Hampshire':2013,'Illinois':2013,
'Maryland':2014,'Minnesota':2014,'New York':2014,'Louisiana':2015,'Arkansas':2016,
'Florida':2016,'Ohio':2016,'North Dakota':2016,'Pennsylvania':2016,'West Virginia':2017,
'Missouri':2018,'Oklahoma':2018,'Utah':2018}
# read panel: Both Sexes, All Ages, All Races-All Origins
rate=defaultdict(dict)
for r in csv.DictReader(open('nchs_state.csv')):
    if r['Sex']!='Both Sexes' or r['Age Group']!='All Ages' or r['Race and Hispanic Origin']!='All Races-All Origins': continue
    st=r['State']; 
    if st=='United States': continue
    try: y=int(r['Year']); v=float(r['Age-adjusted Rate'])
    except: continue
    rate[st][y]=np.log(v)
states=list(rate.keys())
NT=[s for s in states if s not in MML]
print("states:",len(states),"never-treated (no MML by 2019):",len(NT),NT)
YRS=range(1999,2020)
ntser={y:np.mean([rate[s][y] for s in NT if y in rate[s]]) for y in YRS}
# cohorts by effective year (only those with enough pre/post in 1999-2019)
from collections import Counter
treated=[s for s in MML if MML[s]>=2001 and MML[s]<=2016]
coh=defaultdict(list)
for s in treated: coh[MML[s]].append(s)
def es(fs,g,e):
    y=g+e; ref=g-1
    if y<1999 or y>2019 or ref<1999: return None
    d=[rate[s][y]-rate[s][ref] for s in fs if y in rate[s] and ref in rate[s]]
    if not d: return None
    return np.mean(d)-(ntser[y]-ntser[ref])
print(f"\n{'g':>5} {'n':>3} | maxpre(e-4..-1) | post avg(e+1..+4)")
info={}
for g in sorted(coh):
    fs=coh[g]
    pre=[es(fs,g,e) for e in range(-4,0)]; pre=[x for x in pre if x is not None]
    post=[es(fs,g,e) for e in range(1,5)]; post=[x for x in post if x is not None]
    if not pre or not post: continue
    mp=max(abs(x) for x in pre); pa=np.mean(post)
    info[g]=(len(fs),mp,pa)
    print(f"{g:>5} {len(fs):>3} | maxpre={mp:.3f}  pre={['%.2f'%x for x in pre]} | post avg={pa:.3f}  post={['%.2f'%x for x in post]}")
gs=list(info); ns=np.array([info[g][0] for g in gs]); mp=np.array([info[g][1] for g in gs]); pa=np.array([info[g][2] for g in gs])
att=np.average(pa,weights=ns)
print(f"\nPooled ATT (log drug mortality) = {att:.4f}")
for c in [0.10,0.15,0.20,0.25,0.30]:
    keep=mp<=c
    if keep.sum()==0: continue
    latt=np.average(pa[keep],weights=ns[keep])
    print(f"  screen c={c}: keep {keep.sum()}/{len(gs)}  LATT={latt:.4f}  dropped g={[gs[i] for i in range(len(gs)) if not keep[i]]}")
