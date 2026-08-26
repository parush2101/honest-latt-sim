"""
Export the POOLED (all-cohort) aggregated event study and its cluster-bootstrap
covariance, for feeding into the actual HonestDiD R package. Same pipeline as
fracking_figure.py (USDA onset + FHFA county log-HPI, reduced-form vs never-treated,
cluster bootstrap over counties). Reference event time e=-1 (dropped: identically 0).

Writes betahat.csv (event-time, coef), sigma.csv (9x9), meta.json (numPre/numPost, l_vec).
"""
import openpyxl, csv, math, numpy as np, random, json
from collections import defaultdict
random.seed(7); np.random.seed(7)

YRS = list(range(2000, 2012))
rows = list(csv.DictReader(open('shale/usda_oilgas_2000_2011.csv')))
def ser(r, p): return [float(r[f'{p}{y}'] or 0) for y in YRS]
onset = {}; decline = set()
for r in rows:
    f = r['FIPS'].zfill(5)
    if r['oil_gas_change_group'] == 'H_Decline': decline.add(f)
    if r['oil_gas_change_group'] != 'H_Growth': continue
    oil = ser(r, 'oil'); gas = ser(r, 'gas'); boe = [oil[i] + gas[i] / 6 for i in range(len(YRS))]
    peak = max(boe); base = sum(boe[:3]) / 3; thr = base + 0.25 * (peak - base)
    for i, y in enumerate(YRS):
        if boe[i] >= thr and boe[i] > base: onset[f] = y; break

wb = openpyxl.load_workbook('housing/fhfa_county.xlsx', read_only=True); ws = wb.active
hpi = defaultdict(dict)
for i, r in enumerate(ws.iter_rows(values_only=True)):
    if i < 7: continue
    fips = r[2]
    if not (isinstance(fips, str) and len(fips) == 5 and fips.isdigit()): continue
    try: y = int(r[3]); v = float(r[7])
    except: continue
    if v > 0: hpi[fips][y] = math.log(v)

PY = list(range(1998, 2020))
lv = defaultdict(dict); ons = {}
for f in hpi:
    if f in decline: continue
    if any(y not in hpi[f] for y in PY): continue
    for y in PY: lv[f][y] = hpi[f][y]
    ons[f] = onset.get(f, 0)
NT = [f for f in ons if ons[f] == 0]
cohorts = [g for g in sorted(set(ons.values())) if 2003 <= g <= 2011]
byco = {g: [f for f in ons if ons[f] == g] for g in cohorts}

EVENTS = [-5, -4, -3, -2, 0, 1, 2, 3, 4]          # omit reference e=-1 (identically 0)
def att_ge(fs, g, e, nt):
    y = g + e; ref = g - 1
    if y < 1998 or y > 2019: return None
    d = [lv[f][y] - lv[f][ref] for f in fs if y in lv[f]]
    return np.mean(d) - (nt[y] - nt[ref])
def ntser(sampleNT): return {y: np.mean([lv[f][y] for f in sampleNT]) for y in PY}
def eventpath(sm, nt, gs):
    out = {}
    for e in EVENTS:
        vals, wts = [], []
        for g in gs:
            a = att_ge(sm[g], g, e, nt)
            if a is not None: vals.append(a); wts.append(len(sm[g]))
        out[e] = np.average(vals, weights=wts) if vals else np.nan
    return out

nt0 = ntser(NT)
pooled = eventpath(byco, nt0, cohorts)
betahat = np.array([pooled[e] for e in EVENTS])

# cluster bootstrap over counties -> covariance of the pooled event-study vector
B = 2000
draws = np.full((B, len(EVENTS)), np.nan)
for b in range(B):
    sm = {g: [random.choice(byco[g]) for _ in byco[g]] for g in cohorts}
    ntb = [random.choice(NT) for _ in NT]; nt = ntser(ntb)
    pp = eventpath(sm, nt, cohorts)
    draws[b] = [pp[e] for e in EVENTS]
sigma = np.cov(draws, rowvar=False)

numPre = sum(e < 0 for e in EVENTS)               # 4  (e=-5..-2)
numPost = sum(e >= 0 for e in EVENTS)             # 5  (e=0..4)
# target: average post-treatment effect over e=1..4 (0 weight on onset e=0), matching the paper
l_vec = [1.0 if e in (1, 2, 3, 4) else 0.0 for e in EVENTS if e >= 0]
l_vec = [x / sum(l_vec) for x in l_vec]

np.savetxt('housing/betahat.csv', betahat, delimiter=',')
np.savetxt('housing/sigma.csv', sigma, delimiter=',')
json.dump({'events': EVENTS, 'numPre': numPre, 'numPost': numPost, 'l_vec': l_vec,
           'att_e1to4': float(np.array(l_vec) @ betahat[numPre:])},
          open('housing/meta.json', 'w'), indent=2)

print("events   :", EVENTS)
print("betahat  :", np.round(betahat, 4))
print("se (diag):", np.round(np.sqrt(np.diag(sigma)), 4))
print(f"numPre={numPre}  numPost={numPost}  l_vec={np.round(l_vec,3).tolist()}")
print(f"ATT (l'beta_post, e=1..4) = {np.array(l_vec) @ betahat[numPre:]:+.4f}")
print(f"  se(ATT) = {math.sqrt(np.array(l_vec) @ sigma[numPre:, numPre:] @ np.array(l_vec)):.4f}")
print("wrote housing/betahat.csv, housing/sigma.csv, housing/meta.json")
