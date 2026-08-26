import csv
rows=list(csv.DictReader(open('usda_oilgas_2000_2011.csv')))
years=list(range(2000,2012))
def series(r,pref):
    return [float(r[f'{pref}{y}'] or 0) for y in years]
boom=[r for r in rows if r['oil_gas_change_group']=='H_Growth']
print("N H_Growth:",len(boom))
# combined BOE = oil_bbl + gas_mcf/6
cohorts={}
onsets={}
for r in boom:
    oil=series(r,'oil'); gas=series(r,'gas')
    boe=[oil[i]+gas[i]/6.0 for i in range(len(years))]
    peak=max(boe); base=sum(boe[:3])/3  # 2000-2002 baseline
    if peak<=0: continue
    # onset = first year boe exceeds baseline + 25% of (peak-baseline), and rising
    thresh=base+0.25*(peak-base)
    onset=None
    for i,y in enumerate(years):
        if boe[i]>=thresh and boe[i]>base:
            onset=y; break
    onsets[r['FIPS']]=onset
    cohorts[onset]=cohorts.get(onset,0)+1
print("Cohort sizes (onset year -> N counties):")
for y in sorted(cohorts, key=lambda x:(x is None,x)):
    print(f"  {y}: {cohorts[y]}")
# state distribution
from collections import Counter
st=Counter(r['Stabr'] for r in boom)
print("Top states:",st.most_common(12))
