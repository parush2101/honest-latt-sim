# Medical marijuana -> drug overdose mortality (9th app, 2026-08-16) -- NEGATIVE

Data: NCHS "Drug Poisoning Mortality by State, US 1999-2019" (data.cdc.gov 44rk-q6r2),
Both Sexes/All Ages/All Races age-adjusted rate. 51 states, 18 never-treated (no MML by 2019).
Treatment: state medical-marijuana effective year (first-pass coding in probe_mj.py).
Hypothesis: early adopters (pre-fentanyl) protective (Bachhuber 2014); late adopters
swamped by fentanyl -> ATT biased up; flatness screen drops fentanyl-era cohorts -> sign flip.

Result: DOES NOT WORK.
- Pooled ATT = +0.17 (all-drug mortality is fentanyl-dominated over full period).
- Screen keeps 2012-2013 (flat pre-trend, +0.4 post) and DROPS early protective 2004
  (post -0.26 but maxpre 0.32 because n=2 noisy) -> LATT=+0.29, HIGHER than ATT (wrong way).

Two structural reasons (both reconfirm the frontier):
1. Units-per-cohort: 1-5 states/cohort -> pre-trends are noise (maxpre 0.13-0.56);
   screen selects on noise, not credibility. Same barrier as BLL/Castle.
2. Confound not cohort-aligned: fentanyl exposure is GEOGRAPHIC (Appalachia/Northeast),
   not a function of MML adoption year -> no clean early-flat/late-steep split to screen on.

Would need county opioid mortality (CDC WONDER, suppression) AND the confound still
wouldn't align with cohort timing. Not pursued further.
