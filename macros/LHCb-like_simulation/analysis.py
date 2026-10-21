import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.infer.intervals.upper_limits import upper_limit as pyhf_ul
from pyhf.contrib.viz import brazil

import parameters as p
from toy_generator import generate_toy
from models import build_workspace

pyhf.set_backend("jax", "scipy")

def upper_limit(
    seed = p.SEED, 
    cat = [], 
    model = "exp_misid_dseta", 
    calctype = "grid", 
    PLOT = True,
    VERBOSE = True
):

    x, cat_toy, label = generate_toy(seed, exclude_cat = cat) 
    ws = build_workspace(
        x, cat_toy, label, 
        model_type = model, 
        exclude_cat = cat,
        signal_syst = False,
        misid_syst = False, 
        use_staterror = False,
    )
    
    model = ws.model(
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'}
        }
    )
    data = ws.data(model)
    
    kwargs = {"level": p.CL, "return_results": True}
  
    if calctype == "grid":
        print("Calculating Upper Limit with optimized grid scan...")
        scan = np.linspace(0, 5.5, 20)
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=scan, 
            **kwargs
        )
        
    elif calctype == "toms748":
        print("Calculating Upper Limit using Toms748 algorithm (root-finding)...")
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=None, 
            **kwargs
        )
        
    elif calctype == "toybased":
        print("Calculating Upper Limit based on Toy MC...")
        obs, exp, (scan_pts, results) = pyhf_ul(
            data, 
            model, 
            scan=None, 
            calctype="toybased", 
            ntoys=1000,          
            track_progress=True, 
            **kwargs
        )
    else:
        raise ValueError(f"calctype '{calctype}' not recognized.")
    
    Br_obs = p.BR_REF * float(np.asarray(obs).item())
    Br_exp = p.BR_REF * np.array([float(np.asarray(e).item()) for e in exp])
    
    if VERBOSE:
        print(f"\n--- UPPER LIMIT RESULTS ({calctype.upper()}) ---")
        print(f"Br_obs ({p.CL*100:.0f}% CL) = {Br_obs:.3e}")
        print("Br expected:")
        print(f"  median  = {Br_exp[2]:.3e}")
        print(f"  -2σ     = {Br_exp[0]:.3e}")
        print(f"  -1σ     = {Br_exp[1]:.3e}")
        print(f"  +1σ     = {Br_exp[3]:.3e}")
        print(f"  +2σ     = {Br_exp[4]:.3e}\n")
        
    if PLOT and calctype != "toybased":
        fig, ax = plt.subplots()
        brazil.plot_results(scan_pts * p.BR_REF, results, test_size=p.CL, ax=ax)

        ax.set_xlabel("Br", fontsize=12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1.0)
        plt.show()
        
    return Br_obs, Br_exp

if __name__ == "__main__":
    
    #Monte_Carlo and ul distribution
    obs_list = []
    exp_median_list = []
    
    for s in range(p.N_TOYS):
        current_seed = p.SEED + s
        print(f"Running Toy {s+1} of {p.N_TOYS} (Seed: {current_seed})")
        
        try:
            Br_obs, Br_exp = upper_limit(
                seed = current_seed, 
                cat = [], 
                model = "exp_misid_dseta", 
                calctype = "grid",
                PLOT = True, 
                VERBOSE = True
            )
            obs_list.append(Br_obs)
            exp_median_list.append(Br_exp[2])
            
        except Exception as e:
            print(f"Error during toy {s+1} (seed {current_seed}): {e}")
            continue

    if len(obs_list) > 2:        
       
       #Observed    
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.hist(obs_list, bins=15, alpha=0.7, color='dodgerblue', edgecolor='black')
        ax1.set_xlabel("Observed Br Upper Limit", fontsize=12)
        ax1.set_ylabel("Toys", fontsize=12)
        ax1.set_title(f"Observed Upper Limit Distribution ({p.NTOYS} Toys)", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        #Expected
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.hist(exp_median_list, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel("Expected Median Br Upper Limit", fontsize=12)
        ax2.set_ylabel("Toys", fontsize=12)
        ax2.set_title(f"Expected Median Upper Limit Distribution ({p.NTOYS} Toys)", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        
