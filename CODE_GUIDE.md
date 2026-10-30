# Limit setting for $\tau\to\mu\mu\mu$

Codebase for Alessandro Bomei's bachelor thesis (2026): development of an algorithm to compute upper limits on the branching fraction of the ultra-rare decay $\tau\to\mu\mu\mu$ at LHCb, based on [`pyhf`](https://pyhf.readthedocs.io).

The analysis is a toy version of the LHCb Run 2 search. Pseudo-data are generated from the Run 2 fit parameters, written as a HistFactory model (one channel per year and classifier category), and the CLs upper limit on $\mathcal{B}(\tau\to\mu\mu\mu)$ is computed. The limit is then studied as a function of the binning, the categories used, the efficiencies and the toy dataset.

## Pipeline

1. `toy_generator.py` generates pseudo-data for 3 years x 16 categories: combinatorial background (exponential), misidentified $D^+$ and $D_s^+$ decays (Johnson SU), the $D_s\to\eta\mu\nu$ component (template).
2. `models.py` bins the toy and builds the `pyhf` workspace (signal, backgrounds and their systematics).
3. `fast_limit.py` (or `pyhf` itself, see `analysis.py`) fits the model and computes the CLs upper limit, observed and expected.
4. The macros in `macros/` run the different studies and write tables (`data/`) and plots (`plots/`).

## Repository layout

```
.
├── pyproject.toml, uv.lock       # environment (uv)
├── src/mypkg/
│   ├── parameters.py             # ALL parameters and analysis settings
│   ├── toy_generator.py          # generate_toy(): pseudo-data
│   ├── utils.py                  # sampling, binning and plotting functions
│   ├── fast_limit.py             # fast fit and CLs limit (JAX + L-BFGS-B)
│   ├── dseta_extract.py          # one-off: extracts the Ds->eta mu nu pdf
from the ROOT file
│   ├── template_Dsetamunu_Tot.root, template_dseta.npz   # Ds->eta mu nu template
│   └── old_code/                 # first single-bin library, not used anymore
└── macros/
    ├── models.py                 # build_workspace(): pyhf workspace
    ├── analysis.py               # upper_limit(): single limit + Brazil plot
    ├── bins_study.py             # limit vs bin width
    ├── cat_analysis.py           # limit per category and cumulative
    ├── efficiency_grid.py        # limit vs signal/background efficiency
    ├── fit_plots.py              # post-fit plots
    ├── monte_carlo.py            # limit on many toys 
    ├── mc_plots.py               # plots of the Monte Carlo output
    ├── data/                     # text/CSV output
    ├── plots/                    # PDF output
    └── Oldcode/                  # earlier stages of the thesis, not used anymore
```

The macros read and write `data/` and `plots/` relative to the current directory, so run them from `macros/`:

```bash
cd macros
uv run python analysis.py
```

| Macro | What it does | 
|---|---|
| `analysis.py` | Observed and expected limit for one toy, with Brazil plot (see below for the 4 `calctype`) |
| `bins_study.py` | Observed and expected limit vs bin width, relative to the reference width `p.BIN_WIDTH` |
| `cat_analysis.py` | Expected limit of each category alone, then adding categories from the most to the least sensitive |
| `efficiency_grid.py` | Expected limit for a grid of signal and background efficiency factors, as % variation w.r.t. (1, 1) |
| `fit_plots.py` | Fit of one toy, yield table and post-fit plots (`bins`: stacked histograms, `pdf`: smooth curves) |
| `monte_carlo.py` | Observed limit on many toys, in parallel; can inject signal. Resumable: seeds already in the CSV are skipped; `python monte_carlo.py [n_seeds=1000] [first_seed=0] [n_workers] [out.csv]` |
| `mc_plots.py` | Distribution of observed/expected limits, scatter and pull from the Monte Carlo CSV; `python mc_plots.py path/to/file.csv [tag]` |

The toy alone (with its plots) can be generated with `uv run python src/mypkg/toy_generator.py`; plots go to `plots/toy_plots/`.

### Upper limit methods (`analysis.py`)

`upper_limit(seed, cat, model, calctype, grid, ntoys, PLOT, VERBOSE)` returns `(Br_obs, Br_exp)`, with `Br_exp` = [-2s, -1s, median, +1s, +2s].

| `calctype` | Method |
|---|---|
| `fast` | `fast_limit.py`: JAX autodiff + `scipy` L-BFGS-B, asymptotic $\tilde q_\mu$ (Cowan et al.), root finding with `toms748`. Only the median expected limit is filled (the others are `nan`). Used by all the studies |
| `grid` | `pyhf` asymptotic CLs on a grid of $\mu$ (`p.GRID`); needed for the Brazil plot |
| `toms748` | `pyhf` asymptotic CLs, root finding with `toms748` |
| `toybased` | `pyhf` toy-based CLs (`p.N_TOYS` toys) |

## Configuration

Everything is set in `src/mypkg/parameters.py`; the macros follow it. Main switches:

| Parameter | Meaning |
|---|---|
| `MODEL` | `flat` (constant background), `exp` (combinatorial exponential), `exp_misid` (+ $D^+$/$D_s^+$ misidentification), `exp_misid_dseta` (+ $D_s\to\eta\mu\nu$, default) |
| `REGION` | `TOT` (full window, default), `SR` (signal region)|
| `MASS_MIN`, `MASS_MAX`, `SR_BAND` | Mass window [MeV] and half-width of the signal region around $m_\tau$ |
| `BIN_WIDTH` | Bin width [MeV] |
| `CL` | Test size $\alpha$: `0.10` means 90% CL |
| `BR_REF` | Reference branching fraction (`1e-8`): the parameter of interest $\mu$ is in units of `BR_REF`, so $\mathcal{B} = \mu\cdot$`BR_REF` |
| `SEED`, `POI`, `N_TOYS`, `GRID` | Default toy seed, injected $\mu$ in the toy, number of toys for `toybased`, $\mu$ grid for `grid` |
| `EXCLUDE_CAT` | Categories removed from the toy and from the model |
| `SIGNAL_SYST`, `MISID_SYST` | Systematics on the shape parameters of signal and misidentification |
| `CONSTRAIN_YIELD` | Misidentification and $D_s\to\eta\mu\nu$ yields constrained (`normsys`) or free (`normfactor`) |
| `STATERROR` | Template statistical errors (off by default: too many parameters) |
| `EPS_SIG`, `EPS_COMB`, `EPS_DP`, `EPS_DS`, `EPS_DSETA` | Multiplicative factors on signal and background efficiencies |

Fit parameters (yields, slopes, signal and misidentification shapes, efficiencies) come from the LHCb Run 2 fit; masses are from PDG 2024.

## Conventions

- Masses and widths in MeV; $m(\mu\mu\mu)$ window 1606.86 - 1976.86 MeV.
- 16 categories per year, 3 years (2016-2018): 48 channels, named `year_<year>_cat_<category>`.
- The "expected" limit is the median of the Asimov dataset; the "observed" limit is computed on the toy with the chosen seed
- Toys are reproducible: each (year, category, component) uses its own stream from `np.random.SeedSequence(seed)`, so excluding categories does not change the others.
- `analysis.py` applies a small patch to `pyhf`'s NumPy backend (`percentile`) for compatibility with recent NumPy.

## Credits

Fit parameters, `limit.cpp` and the $D_s\to\eta\mu\nu$ template: Davide Fazzini (LHCb $\tau\to\mu\mu\mu$ Run 2 analysis).
