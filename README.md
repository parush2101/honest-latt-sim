# Trading Scope for Credibility in Difference-in-Differences

Replication code, data, and manuscript for *Trading Scope for Credibility in
Difference-in-Differences* (Arora & Chand).

## The idea

In staggered difference-in-differences, parallel trends may hold for some treated
cohorts and fail for others, so the ATT is biased and honest inference that keeps
the ATT as the target is either wide or falsely narrow. We instead **change the
estimand** to the credible-subpopulation LATT — the effect for the cohorts whose
parallel trends is credible — and make the data-driven selection **honest** by
composing post-selection carving (Lee et al. 2016) with the sensitivity bounds of
Rambachan & Roth (2023).

## Repository layout

```
paper.tex, refs.bib, paper.pdf   The manuscript (build from repo root).
figures/                          Figures included by the paper (+ m_sweep supporting).
code/                             Simulation scripts (Sections 5, Appendix A).
application/
    shale/                        USDA oil-and-gas onset data + panel build.
    housing/                      FHFA county HPI, event-study export, app scripts, results.
results/                          Analysis notes (e.g. the Theorem 2 derivation).
archive/                          Superseded scripts and earlier exploration (not used by the paper).
```

## Build the paper

From the repo root:

```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Figures are resolved from `figures/` via `\graphicspath`.

## Reproduce the exhibits

Simulation figures/tables (run from `code/`; each writes its PNG to the current
directory — copy the output into `figures/`):

| Exhibit | Script |
|---|---|
| Table 2 (estimand demonstration) | `code/tier1.py` |
| Figure 1 (scope condition) | `code/master_axis.py` |
| Figure 2 (honest coverage) | `code/layer2_full.py` |
| Figure 3 / Table 4 (width dominance) | `code/width_dominance.py` |
| Table 5 (carved coverage across γ) | `code/carved_gamma.py` |
| Figure 5 / Table (optimal weighting) | `code/prop11.py` |
| §5.4 sensitivity numbers | `code/m_sweep.py` |

`carved_gamma.py` imports the exact polyhedral carving core from `race2.py` /
`race_multipre.py` (same folder).

Application (run from `application/`):

```bash
cd application
python3 housing/fracking_figure.py      # Figure 4 -> fracking_diagnostic.png
python3 housing/carved_application.py    # §6 carved procedure + Step-5 numbers (Gamma*)
```

`carved_application.py` estimates Σ by cluster bootstrap (the estimated-Σ case),
runs the randomized carved interval on the retained cohorts, and reports the
selection cell, carved endpoints, and the composition-gap breakdown Γ*.

## Data notes

Large raw inputs are git-ignored and re-downloaded from public sources: the FHFA
county house-price index (`application/housing/fhfa_county.xlsx`) and the built
shale panel (`application/shale/shale_panel.csv`, produced by `build_shale_panel.py`).
The committed USDA onset file drives cohort assignment.
