'''
cat_analysis.py

I study how the UL changes considering:
1) Single category
2) Cumulative category per sensitivity

'''

import os 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

import mypkg.parameters as p
from mypkg.toy_generator import generate_toy
from models import build_workspace
import mypkg.fast_limit as fl
from analysis import upper_limit  

MODEL_TYPE = p.MODEL
PLOTS_DIR = os.path.join("plots", "categories")
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "categories_analysis.txt")
LOW_STAT = 20 #I checi if the number of events is enough to use asymptotics

def main():
    os.makedirs(PLOTS_DIR, exist_ok = True)
    log_f = open(LOG_FILE, "w", encoding = "utf-8")
    
    arg = sys.argv[1] if len(sys.argv) > 1 else str(p.SEED)
    seed = None if arg.lower() == "none" else int(arg)
    
    def log_print(text=""):
        print(text, flush = True)
        log_f.write(text + "\n")
        log_f.flush()
    
    #workspace
    x, cat, label = generate_toy(p.SEED, verbose = False)
    ws = build_workspace(x, cat, label, model_type = "exp_misid_dseta")
    all_names = [ch["name"] for ch in ws["channels"]]
    n_cat = p.CATEGORIES

    def sub_model(cats):
        keep = {f"year_{y}_cat_{c}" for y in p.YEARS for c in cats}
        drop = [n for n in all_names if n not in keep]
        return (ws.prune(channels = drop) if drop else ws).model()
        
    #Expected bkg and signal in the SR for each category from workspace
    n_bins = int(round((p.MASS_MAX - p.MASS_MIN) / p.BIN_WIDTH))
    edges = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
    in_sr = (0.5 * (edges[1:] + edges[:-1]) >= p.SR_MIN) & (0.5 * (edges[1:] + edges[:-1]) <= p.SR_MAX)
    b_sr, s_sr = np.zeros(n_cat), np.zeros(n_cat)
    for ch in ws["channels"]:
        c = int(ch["name"].split("_")[-1])
        for s in ch["samples"]:
            counts = np.array(s["data"])[in_sr].sum()
            if s["name"] == "signal":
                s_sr[c] += counts
            else:
                b_sr[c] += counts

    log_print(f"Model '{MODEL_TYPE}', {len(all_names)} channels, {n_cat} categories.")
    log_print(f"Limits in BR unit (BR_REF = {p.BR_REF:g}), CL = {(1 - p.CL) * 100:.0f}%\n")
    
    #single_category analysis
    log_print("1. Single category analysis (ordered for sensibility):")
    
    t0 = time.time()
    single_limits = np.zeros(n_cat)
    for c in range(n_cat):
        m = sub_model([c])
        ctx = fl.setup(m)
        single_limits[c] = fl.expected_limit(ctx, level = p.CL)
    
    order = list(np.argsort(single_limits))
    log_print("1. Single categories (years combined), ordered by sensitivity:")
    log_print(f"{'rank':>4} {'cat':>4} | {'expected UL':>10} | {'SR bkg':>9} {'SR sig':>11} | note")
    for r, c in enumerate(order, 1):
        note = "toy?" if b_sr[c] < LOW_STAT else ""
        log_print(f"{r:4d} {c:4d} | {single_limits[c] * p.BR_REF:10.3e} | {b_sr[c]:9.1f} {s_sr[c] * single_limits[c]:11.2f} | {note}")
    log_print(f"(SR sig = expected signal events in SR at category expected limit; note: SR bkg < {LOW_STAT}; {time.time() - t0:.0f} s)\n")

    #bunches of categories
    log_print("\n2. Cumulative curve (adding more sensitive categories):")
    log_print(f"{'N':>3} {'Added':>9} | {'Expected UL (Median)':>20} {'vs N-1':>8} | {'Observed UL':>20}")
    
    cum_exp = np.zeros(n_cat)
    cum_obs = np.full(n_cat, np.nan)
    
    for n in range(1, n_cat + 1):
        cats_to_include = order[:n]
        excluded_cats = [c for c in range(n_cat) if c not in cats_to_include]
        
        seed = p.SEED if seed is None else seed
        Br_obs, Br_exp = upper_limit(
            seed=seed,
            cat=excluded_cats,
            model=MODEL_TYPE,
            calctype="fast",
            PLOT=False,
            VERBOSE=False
        )
        
        cum_exp[n - 1] = Br_exp[2]  
        if seed is not None:
            cum_obs[n - 1] = Br_obs 
        
        # Percentage gain compared to N-1
        gain = f"{(cum_exp[n - 1] / cum_exp[n - 2] - 1) * 100:+7.1f}%" if n > 1 else ""
        obs_str = f"{cum_obs[n - 1]:20.3e}" if seed is not None else f"{'':>20}"
        
        log_print(f"{n:3d} {order[n - 1]:9d} | {cum_exp[n - 1]:20.3e} {gain:>8} | {obs_str}")
        
    # Plots    
    with plt.style.context(hep.style.LHCb2 if 'hep' in globals() else 'default'):
        # Plot 1: Single category
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(n_cat), single_limits[order] * p.BR_REF, color="#4f81bd", edgecolor="black", linewidth=0.5)
        for i, c in enumerate(order):
            if b_sr[c] < LOW_STAT:
                ax.text(i, single_limits[c] * p.BR_REF, "toy?", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(n_cat))
        ax.set_xticklabels([str(c) for c in order])
        ax.set_xlabel("Category (ordered by sensitivity)")
        ax.set_ylabel(rf"Expected UL ({(1 - p.CL) * 100:.0f}% CL), single category")
        fig.savefig(os.path.join(PLOTS_DIR, "categories_single.pdf"), bbox_inches="tight")
        plt.close(fig)

        # Plot 2: Cumulative curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, n_cat + 1), cum_exp, "o-", color="#4f81bd", label="expected (Asimov)")
        if seed is not None:
            ax.plot(range(1, n_cat + 1), cum_obs, "s--", color="#c0504d", label=f"observed (toy, seed {seed})")
        ax.set_xlabel("Number of categories (most sensitive ones)")
        ax.set_ylabel(rf"UL ({(1 - p.CL) * 100:.0f}% CL)")
        ax.set_xticks(range(1, n_cat + 1))
        ax.legend(frameon=False)
        fig.savefig(os.path.join(PLOTS_DIR, "categories_cumulative.pdf"), bbox_inches="tight")
        plt.close(fig)

    log_f.close()
    print(f"Report saved in: {LOG_FILE}")
    print(f"Plots saved in: {PLOTS_DIR}/")

if __name__ == "__main__":
    main()

