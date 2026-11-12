'''
efficiency.py

Computes the UL varying the efficiency of signal and bkg
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mplhep as hep

import mypkg.parameters as p
import mypkg.fast_limit as fl
from mypkg.toy_generator import generate_toy
from models import build_workspace

MODEL_TYPE = p.MODEL
EXCLUDE_CAT = [int(c) for c in os.environ.get("EXCLUDE_CAT", "").split(",") if c.strip() != ""]
EPS_SIG = [float(v) for v in os.environ.get("EFF_SIG_LIST", "1,0.80,0.60,0.40, 0.20").split(",")]
EPS_BKG = [float(v) for v in os.environ.get("EFF_BKG_LIST", "1,0.80,0.60,0.40, 0.20").split(",")]
ALL_BKG = os.environ.get("ALL_BKG", "0") == "1"

DATA_DIR = "data"
PLOTS_DIR = os.path.join("plots", "efficiency")
os.makedirs(DATA_DIR, exist_ok = True)
os.makedirs(PLOTS_DIR, exist_ok = True)
log_path = os.path.join(DATA_DIR, "efficiency_grid.txt")

log_f = open(log_path, "w")
def log_print(text = ""):
    print(text, flush = True)
    log_f.write(text + "\n")
    log_f.flush()

x, cat, label = generate_toy(p.SEED, exclude_cat = EXCLUDE_CAT, verbose = False)

log_print(f"Median expected limit (Asimov), {(1 - p.CL) * 100:.0f}% CL, in units of BR (BR_REF = {p.BR_REF:g}); eps_bkg factor applied to: {'all background' if ALL_BKG else 'combinatorial only'}")
ul = np.zeros((len(EPS_SIG), len(EPS_BKG)))

for i, es in enumerate(EPS_SIG):
    for j, eb in enumerate(EPS_BKG):
        t = time.time()
        model = build_workspace(
            x, cat, label, 
            model_type = MODEL_TYPE, 
            exclude_cat = EXCLUDE_CAT,
            eps_sig = es,
            eps_comb = eb,
            eps_dp = eb if ALL_BKG else 1.0,
            eps_ds = eb if ALL_BKG else 1.0,
            eps_dseta = eb if ALL_BKG else 1.0
        ).model()
        
        ul[i, j] = fl.expected_limit(fl.setup(model), level = p.CL)
        print(f"  eps_sig = {es:.2f}, eps_bkg = {eb:.2f}: {ul[i, j] * p.BR_REF:.3e}  [{time.time() - t:.0f} s]", flush = True)

ref = ul[EPS_SIG.index(1.0), EPS_BKG.index(1.0)] if (1.0 in EPS_SIG and 1.0 in EPS_BKG) else ul[0, 0]
log_print("\nPercentage variation of expected limit relative to (eps_sig, eps_bkg) = (1, 1) (negative = improvement)")
log_print(f"{'eps_sig \\ eps_bkg':>18} " + " ".join(f"{eb:8.2f}" for eb in EPS_BKG))
for i, es in enumerate(EPS_SIG):
    log_print(f"{es:18.2f} " + " ".join(f"{(ul[i, j] / ref - 1) * 100:+7.1f}%" for j in range(len(EPS_BKG))))
log_print(f"\n(reference: expected UL = {ref * p.BR_REF:.3e})")
best = np.unravel_index(np.argmin(ul), ul.shape)
log_print(f"Grid minimum: eps_sig = {EPS_SIG[best[0]]:.2f}, eps_bkg = {EPS_BKG[best[1]]:.2f}, expected UL = {ul[best] * p.BR_REF:.3e} ({(ul[best] / ref - 1) * 100:+.1f}%)")

plot_filename = "efficiency_grid.pdf"
plot_filepath = os.path.join(PLOTS_DIR, plot_filename)

with plt.style.context(hep.style.LHCb2):
    fig, ax = plt.subplots(figsize = (9, 6))
    ratio = ul / ref
    logr = ul
    norm = TwoSlopeNorm(vmin = logr.min(), vcenter = 0, vmax = logr.max())
    im = ax.imshow(logr, origin = "lower", cmap = "RdBu_r", norm = norm)
    for i in range(len(EPS_SIG)):
        for j in range(len(EPS_BKG)):
            r, g, b, _ = im.cmap(im.norm(logr[i, j]))
            col = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
            ax.text(j, i, f"{(ratio[i, j] - 1) * 100:+.0f}%", ha = "center", va = "center", fontsize = 16, color = col)
    ax.set_xticks(range(len(EPS_BKG)))
    ax.set_xticklabels([f"{v:g}" for v in EPS_BKG])
    ax.set_yticks(range(len(EPS_SIG)))
    ax.set_yticklabels([f"{v:g}" for v in EPS_SIG])
    ax.set_xlabel(r"$\varepsilon_{bkg}$ (combinatorial)" if not ALL_BKG else r"$\varepsilon_{bkg}$ (all background)")
    ax.set_ylabel(r"$\varepsilon_{sig}$")
    cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04, label = "UL / UL(1,1)")
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 5) if ratio.min() <= t <= ratio.max()]
    cbar.set_ticks(np.log(ticks))
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    fig.savefig(plot_filepath, bbox_inches = "tight")
    plt.close(fig)
    
log_print(f"\nPlot saved: {plot_filepath}; table in {log_path}")
log_f.close()
