'''
toy.generator.py

Data are generated according to Poisson of yield for category.

Background is composed of three components:
- Exponential (Slope from limit.cpp)
- MisId for Dp and Ds (Johnson SU, parameters from limit.cpp)
- Ds_eta (template from Davide Fazzini)

'''
import os
import numpy as np
import mypkg.parameters as p
from mypkg.utils import sample_combinatorial, sample_johnson, sample_Ds, sample_Dp, sample_Dseta, plot_toy

rng = np.random.default_rng(p.SEED)
    
def generate_toy(
    seed = p.SEED, 
    POI = p.POI, 
    include_signal = False,
    lo = p.MASS_MIN,
    hi = p.MASS_MAX,
    MisId = True,
    Dseta = True,
    plot = False,
    plot_year = 0,
    output_dir = os.path.join("plots", "toy_plots"),
    exclude_cat = p.EXCLUDE_CAT,
    verbose = False
):
    children = np.random.SeedSequence(seed).spawn(len(p.YEARS) * p.CATEGORIES * p.N_BKG_COMP)
    def stream(ic, k):
        return np.random.default_rng(children[ic * p.N_BKG_COMP + k])
    
    all_x, all_cat, all_label = [], [], []
    
    exp_yields = {
        "bkg_combinatorial": 0.0,
        "bkg_Dp": 0.0,
        "bkg_Ds": 0.0,
        "bkg_Dseta": 0.0,
        "sig": 0.0
    }   
    for iy, y in enumerate(p.YEARS):
        for c in range(p.CATEGORIES):
            ic = iy * p.CATEGORIES + c   
            
            if c in exclude_cat:
                continue
                
            #combinatorial background
            rng = stream(ic, 0)
            exp_comb = p.C_BKG_YIELDS[y][c]
            exp_yields["bkg_combinatorial"] += exp_comb
            n_bkg_combinatorial = rng.poisson(p.C_BKG_YIELDS[y][c])
            bkg_combinatorial = sample_combinatorial(rng, n_bkg_combinatorial, p.SLOPE_BKG_VALS[y][c])
            all_x.append(bkg_combinatorial)
            all_cat.append(np.full(n_bkg_combinatorial, ic))
            all_label.append(np.full(n_bkg_combinatorial, "bkg_combinatorial"))
            
            if MisId:
                #MisId Dp3pi
                rng = stream(ic, 1)                
                exp_dp = p.DP_YIELDS[y] * p.DP_EFF[y][c]
                exp_yields["bkg_Dp"] += exp_dp
                n_bkg_Dp = rng.poisson(exp_dp)
                bkg_Dp   = sample_Dp(rng, n_bkg_Dp, y)
                all_x.append(bkg_Dp)
                all_cat.append(np.full(n_bkg_Dp, ic))
                all_label.append(np.full(n_bkg_Dp, "bkg_Dp"))
                
                #MisId Ds3pi
                rng = stream(ic, 2)
                exp_ds = p.DS_YIELDS[y] * p.DS_EFF[y][c]
                exp_yields["bkg_Ds"] += exp_ds
                n_bkg_Ds = rng.poisson(exp_ds)
                bkg_Ds   = sample_Ds(rng, n_bkg_Ds, y)
                all_x.append(bkg_Ds)
                all_cat.append(np.full(n_bkg_Ds, ic))
                all_label.append(np.full(n_bkg_Ds, "bkg_Ds"))
            
            if Dseta:
                #Dseta
                rng = stream(ic, 3)
                exp_dseta = p.DSETA_YIELDS[y] * p.DSETA_EFF[y][c]
                exp_yields["bkg_Dseta"] += exp_dseta
                n_bkg_Dseta = rng.poisson(exp_dseta)
                bkg_Dseta   = sample_Dseta(rng, n_bkg_Dseta)
                all_x.append(bkg_Dseta)
                all_cat.append(np.full(n_bkg_Dseta, ic))
                all_label.append(np.full(n_bkg_Dseta, "bkg_Dseta"))

    x = np.concatenate(all_x) if all_x else np.empty(0)
    cat = np.concatenate(all_cat) if all_cat else np.empty(0, dtype=int)
    label = np.concatenate(all_label) if all_label else np.empty(0, dtype="U20")            
    
    if x.size and (x.min() < p.MASS_MIN or x.max() > p.MASS_MAX):
        raise ValueError('events outside mass window')
    in_sr = (x >= p.SR_MIN) & (x <= p.SR_MAX)
    
    if verbose:
        print("\n================ GENERATED TOY STATISTICS ================")
        print(f"{'Component':<22} | {'Expected (Tot)':<15} | {'Observed (Tot)':<16} | {'Observed (SR)':<15}")
        print("-" * 75)
        for comp in ["bkg_combinatorial", "bkg_Dp", "bkg_Ds", "bkg_Dseta", "sig"]:
            if not include_signal and comp == "sig":
                continue
            exp_val = exp_yields[comp]
            obs_tot = (label == comp).sum()
            obs_sr = ((label == comp) & in_sr).sum()
            print(f"{comp:<22} | {exp_val:<15.1f} | {obs_tot:<16d} | {obs_sr:<15d}")
        print("-" * 75)
        print(f"Total Generated Events (Full Window) : {len(x)}")
        print(f"Total Events in Signal Region (SR)  : {in_sr.sum()}")
        print("==========================================================\n")
 
    if plot:
        plot_toy(x, cat, label, include_signal=include_signal, output_dir=output_dir)
 
    return x, cat, label
 
if __name__ == "__main__":
    generate_toy(seed=p.SEED, POI=p.POI, include_signal=False, plot=True, verbose=True)
    
    
 
