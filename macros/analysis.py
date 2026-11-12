'''
analysis.py

Compute Upper limit and return Brazil plot.
Four calctype are possible:
1) Grid
2) Toms748
3) Toy
4) Fast limit (see fast_limit.py for better details)
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.infer.intervals.upper_limits import upper_limit as pyhf_ul
from pyhf.contrib.viz import brazil

import mypkg.parameters as p
from mypkg.toy_generator import generate_toy
from models import build_workspace

from pyhf.tensor.numpy_backend import numpy_backend 
import mypkg.fast_limit as fl
import mplhep as hep

import jax.numpy as jnp
from pyhf.tensor.jax_backend import jax_backend

PLOTS_DIR = os.path.join("plots", "upper_limit")

numpy_backend.percentile = lambda self, t, q, axis = None, interpolation = "linear": \
    np.percentile(t, q, axis = axis, method = interpolation) #fix for pyhf

jax_backend.percentile = lambda self, t, q, axis = None, interpolation = "linear": \
    jnp.percentile(t, q, axis = axis, method = interpolation) #fix for jax    

def upper_limit(
    seed = p.SEED, 
    cat = [], 
    model = p.MODEL, 
    calctype = "fast", 
    grid = p.GRID,
    ntoys = p.N_TOYS,
    PLOT = True,
    VERBOSE = True,
):
    
    pyhf.set_backend("numpy" if calctype == "toms748" else "jax", "scipy")
    
    x, cat_toy, label = generate_toy(seed, exclude_cat = cat) 
    ws = build_workspace(
        x, cat_toy, label, 
        model_type = model, 
        exclude_cat = cat,
        signal_syst = p.SIGNAL_SYST,
        misid_syst = p.MISID_SYST, 
        use_staterror = p.STATERROR,
    )
    
    model = ws.model(
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'}
        }
    )
    
    #I extract the datas without rebuilding the workspace each time
    data = fl.counts(model, x, cat_toy)
    
    kwargs = {"level": p.CL, "return_results": True}
    scan_pts = None
    results = None
  
  
    if calctype == "grid":
        if VERBOSE:
            print("Calculating Upper Limit with grid scan...")
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=grid, 
            **kwargs
        )
        
        Br_obs = p.BR_REF * float(np.asarray(obs).item())
        Br_exp = p.BR_REF * np.array([float(np.asarray(e).item()) for e in exp])
        
        
    elif calctype == "toms748":
        if VERBOSE:
            print("Calculating Upper Limit using Toms748 algorithm...")
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=None, 
            **kwargs
        )
        
        Br_obs = p.BR_REF * float(np.asarray(obs).item())
        Br_exp = p.BR_REF * np.array([float(np.asarray(e).item()) for e in exp])
        
    elif calctype == "toybased":
        if VERBOSE:
            print("Calculating Upper Limit based on Toy MC...")
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=grid, 
            calctype="toybased", 
            ntoys=ntoys,          
            track_progress=True, 
            **kwargs
        )

    elif calctype == "fast":
        if VERBOSE:
            print("Calculating Upper Limit using FAST approach (L-BFGS-B + JAX)...")
        ctx = fl.setup(model)
        limit_obs, info_obs = fl.upper_limit_obs(ctx, data, level = p.CL)
        limit_exp = fl.expected_limit(ctx, level = p.CL)

    else:
        raise ValueError(f"calctype '{calctype}' not recognized.")
    
    if calctype == "fast":
        Br_obs = p.BR_REF * float(limit_obs)
        Br_exp_med = p.BR_REF * float(limit_exp)
        Br_exp = np.array([np.nan, np.nan, Br_exp_med, np.nan, np.nan])  
    else:
        Br_obs = p.BR_REF * float(np.asarray(obs).item())
        Br_exp = p.BR_REF * np.array([float(np.asarray(e).item()) for e in exp])
    
    if VERBOSE:
        print(f"\n--- UPPER LIMIT RESULTS ({calctype.upper()}) ---")
        print(f"Br_obs ({1-p.CL*100:.0f}% CL) = {Br_obs:.3e}")
        print("Br expected:")
        print(f"  median  = {Br_exp[2]:.3e}")
        if not np.isnan(Br_exp[0]):
            print(f"  -2σ     = {Br_exp[0]:.3e}")
            print(f"  -1σ     = {Br_exp[1]:.3e}")
            print(f"  +1σ     = {Br_exp[3]:.3e}")
            print(f"  +2σ     = {Br_exp[4]:.3e}\n")
        else:
            print("  (±1σ/±2σ omitted in FAST)\n")
        
    if PLOT and scan_pts is not None and results is not None:
        UNIT = 1e-9
        with plt.style.context(hep.style.LHCb2):
            plt.rcParams["mathtext.fontset"] = "stix"
            fig, ax = plt.subplots(figsize = (9, 6), layout = "constrained")
            artists = brazil.plot_results(
                scan_pts * p.BR_REF / UNIT,
                results,
                test_size = p.CL,
                ax = ax
            )

            artists.test_size.remove()
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 1, 3, 2] # CLs, CLs_exp, ±1σ, ±2σ
            ax.legend([handles[i] for i in order], [labels[i] for i in order], loc = "upper right", frameon = True, fancybox = False, edgecolor = "black", framealpha = 1)

            ul_max = max(Br_obs, Br_exp[2]) / UNIT
            ax.hlines(p.CL, 0, ul_max, color = "red", lw = 1.2)
            ax.vlines([Br_obs / UNIT, Br_exp[2] / UNIT], 0, p.CL, color = "red", lw = 1.2)

            ax.set_xlim(0, 50)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlabel(r"$B(\tau^- \to \mu^-\mu^+\mu^-)$ [$10^{-9}$]")

            os.makedirs(PLOTS_DIR, exist_ok = True)
            plot_path = os.path.join(PLOTS_DIR, f"brazil_plot_{calctype}.pdf")
            fig.savefig(plot_path)
        plt.show()
        
    return Br_obs, Br_exp

if __name__ == "__main__":
    upper_limit(
        seed = p.SEED, 
        cat = [], 
        model = p.MODEL, 
        calctype = "grid", 
        grid = p.GRID,
        ntoys = p.N_TOYS,
        PLOT = True,
        VERBOSE = True,
    )
