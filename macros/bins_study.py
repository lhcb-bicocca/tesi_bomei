'''
bins_study.py

I study the UL considering different mass bins width

'''

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import mplhep as hep

import mypkg.parameters as p
from analysis import upper_limit                

MODEL_TYPE = p.MODEL
EXCLUDE_CAT = [int(c) for c in os.environ.get("EXCLUDE_CAT", "").split(",") if c.strip() != ""]
WIDTHS = [float(w) for w in os.environ.get("WIDTHS", "74,37,20,15,10,5").split(",")]
REF = p.BIN_WIDTH if p.BIN_WIDTH in WIDTHS else WIDTHS[0]
CALCTYPE = "fast"

PLOTS_DIR = os.path.join("plots", "bin_plot")
DATA_DIR = "data"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
first = int(sys.argv[2]) if len(sys.argv) > 2 else 42
seeds = list(range(first, first + n_seeds))

log_path = os.path.join(DATA_DIR, "bins_study.txt")
log_f = open(log_path, "w", encoding = "utf-8")
def log_print(text = ""):
    print(text, flush = True)
    log_f.write(text + "\n")
    log_f.flush()

log_print(f"Model '{MODEL_TYPE}', excluded categories = {EXCLUDE_CAT}, {n_seeds} toys (seed {seeds[0]}..{seeds[-1]}), calctype = {CALCTYPE}")
log_print(f"Limits in units of BR (BR_REF = {p.BR_REF:g}), CL = {(1 - p.CL) * 100:.0f}%; reference = {REF:g} MeV\n")

expected, observed, n_bins_of, width_of = {}, {}, {}, {}
for w in WIDTHS:
    t = time.time()
    p.BIN_WIDTH = w                                  
    n_bins = int(round((p.MASS_MAX - p.MASS_MIN) / w))
    n_bins_of[w], width_of[w] = n_bins, (p.MASS_MAX - p.MASS_MIN) / n_bins
    
    # Expected_limit
    _, Br_exp_arr = upper_limit(
        seed = seeds[0], 
        cat = EXCLUDE_CAT, 
        model = MODEL_TYPE, 
        calctype = CALCTYPE, 
        PLOT = False, 
        VERBOSE = False
    )
    expected[w] = Br_exp_arr[2] 
    
    # Observed limits
    obs_list = []
    for s in seeds:
        Br_obs, _ = upper_limit(
            seed = s, 
            cat = EXCLUDE_CAT, 
            model = MODEL_TYPE, 
            calctype = CALCTYPE, 
            PLOT = False, 
            VERBOSE = False
        )
        obs_list.append(Br_obs)
    observed[w] = np.array(obs_list)
    
    log_print(f"  w = {w:5.1f} MeV ({n_bins:3d} bins of {width_of[w]:.2f} MeV): expected {expected[w]:.3e}, observed (mean) {observed[w].mean():.3e}  [{time.time() - t:.0f} s]")

#Summary table
log_print(f"\n{'width':>10} {'bins':>4} | {'expected UL':>10} {'vs ref.':>8} | {'observed UL (mean)':>20} | {'paired ratio UL(w)/UL(ref.)':>34}")
for w in WIDTHS:
    ratio = observed[w] / observed[REF]
    err = ratio.std(ddof = 1) / np.sqrt(len(ratio)) if len(ratio) > 1 else np.nan
    pair = f"{ratio.mean():.4f} +/- {err:.4f}" if len(ratio) > 1 else f"{ratio.mean():.4f}"
    log_print(f"{width_of[w]:10.2f} {n_bins_of[w]:4d} | {expected[w]:10.3e} {(expected[w] / expected[REF] - 1) * 100:+7.1f}% | {observed[w].mean():20.3e} | {pair:>34}")
log_print("(width = actual bin width; expected 'vs ref.' = percentage variation relative to the reference width)")

# Plot
with plt.style.context(hep.style.LHCb2):
    fig, ax = plt.subplots(figsize = (9, 6))
    ws = np.array([width_of[w] for w in WIDTHS])
    ax.plot(ws, [expected[w] / expected[REF] for w in WIDTHS], "o-", color = "#4f81bd", label = "expected (Asimov)")
    if n_seeds > 1:
        rat = [(observed[w] / observed[REF]).mean() for w in WIDTHS]
        err = [(observed[w] / observed[REF]).std(ddof = 1) / np.sqrt(n_seeds) for w in WIDTHS]
        ax.errorbar(ws, rat, yerr = err, fmt = "s--", color = "#c0504d", label = f"observed, paired ({n_seeds} toys)")
    else:
        ax.plot(ws, [observed[w][0] / observed[REF][0] for w in WIDTHS], "s--", color = "#c0504d", label = f"observed (seed {seeds[0]})")
    ax.axhline(1.0, color = "gray", lw = 0.8)
    ax.set_xscale("log")
    
    xt = np.sort(ws)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{w:g}" for w in sorted(WIDTHS)])
    ax.tick_params(axis = "x", which = "major", labelsize = 30 )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.margins(x = 0.08, y = 0.08)
    
    ax.set_xlabel("Bin width [MeV]")
    ax.set_ylabel(f"UL / UL({REF:g} MeV)")
    ax.legend(frameon = False)
    plot_path = os.path.join(PLOTS_DIR, "bins_study.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

log_print("\nPlot saved: bins_study.pdf; table in data/bins_study.txt")
log_f.close()
