'''
toy.generator.py
(It is possible to include signal (sampled from a johnson SU), but I keep it deactivated).

Data are generated according to Poisson of yield for category.

Background is composed of three components:
- Exponential (Slope from limit.cpp)
- MisId for Dp and Ds (Johnson SU, parameters from limit.cpp)
- Ds_eta (template from Fazzini)

'''
import os
import numpy as np
import parameters as p
from scipy.stats import johnsonsu
import matplotlib.pyplot as plt

rng = np.random.default_rng(p.SEED)

def sample_combinatorial(rng, n, slope):
    if n == 0:
        return np.array([])
    
    u = rng.uniform(0, 1, size=n)
    # if slope ~ 0, I use flat distribution
    if abs(slope) < 1e-10:
        return rng.uniform(p.MASS_MIN, p.MASS_MAX, size=n)
    
    exp_min = np.exp(slope * p.MASS_MIN)
    exp_max = np.exp(slope * p.MASS_MAX)
    
    x = (1.0 / slope) * np.log(exp_min + u * (exp_max - exp_min))
    return x

def sample_johnson(rng, n, year, loc, scale):
    if n == 0:
        return np.array([])
 
    d = johnsonsu(p.MISID_TAIL_Q0[year], p.MISID_TAIL_Q1[year], loc=loc, scale=scale)
    u = rng.uniform(d.cdf(p.MASS_MIN), d.cdf(p.MASS_MAX), size=n)
    return d.ppf(u)
 
def sample_Ds(rng, n, year):
    return sample_johnson(rng, n, year, p.DS_MEAN_CALIB[year], p.DS_SIGMA[year])
 
def sample_Dp(rng, n, year):
    return sample_johnson(rng, n, year, p.DP_MEAN_CALIB[year], p.DP_SIGMA[year])
    
data = np.load(p.DSETA_TEMPLATE)
xs = data["x"]
ys = data["pdf"]
 
dx = np.diff(xs)
cdf = np.concatenate([[0], np.cumsum(0.5 * (ys[1:] + ys[:-1]) * dx)])
cdf /= cdf[-1]

def sample_Dseta(rng, n):
    if n == 0:
        return np.array([])
 
    c_lo = np.interp(p.MASS_MIN, xs, cdf)
    c_hi = np.interp(p.MASS_MAX, xs, cdf)
    u = rng.uniform(c_lo, c_hi, size=n)
    return np.interp(u, cdf, xs)
	
'''	
def sample_Dseta(n, year):

    if n == 0:
        return np.array([])
    
    return 
'''
    
def generate_toy(
    seed = p.SEED, 
    POI = p.POI, 
    lo = p.MASS_MIN,
    hi = p.MASS_MAX,
    include_signal = False,
    MisId = True,
    Dseta = True,
    plot = False,
    plot_year = 0,
    output_dir = "toy_plots",
    verbose = True
):
    rng = np.random.default_rng(seed)
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
            
            if c == 0:
                continue
                
            #combinatorial background
            exp_comb = p.C_BKG_YIELDS[y][c]
            exp_yields["bkg_combinatorial"] += exp_comb
            n_bkg_combinatorial = rng.poisson(p.C_BKG_YIELDS[y][c])
            bkg_combinatorial = sample_combinatorial(rng, n_bkg_combinatorial, p.SLOPE_BKG_VALS[y][c])
            all_x.append(bkg_combinatorial)
            all_cat.append(np.full(n_bkg_combinatorial, ic))
            all_label.append(np.full(n_bkg_combinatorial, "bkg_combinatorial"))
            
            if MisId:
                #MisId Dp3pi                
                exp_dp = p.DP_YIELDS[y] * p.DP_EFF[y][c]
                exp_yields["bkg_Dp"] += exp_dp
                n_bkg_Dp = rng.poisson(exp_dp)
                bkg_Dp   = sample_Dp(rng, n_bkg_Dp, y)
                all_x.append(bkg_Dp)
                all_cat.append(np.full(n_bkg_Dp, ic))
                all_label.append(np.full(n_bkg_Dp, "bkg_Dp"))
                
                #MisId Ds3pi
                exp_ds = p.DS_YIELDS[y] * p.DS_EFF[y][c]
                exp_yields["bkg_Ds"] += exp_ds
                n_bkg_Ds = rng.poisson(exp_ds)
                bkg_Ds   = sample_Ds(rng, n_bkg_Ds, y)
                all_x.append(bkg_Ds)
                all_cat.append(np.full(n_bkg_Ds, ic))
                all_label.append(np.full(n_bkg_Ds, "bkg_Ds"))
            
            if Dseta:
                #Dseta
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
        # Create the plots directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        bin_width = p.BIN_WIDTH
        n_bins = int(round((p.MASS_MAX - p.MASS_MIN) / bin_width))
        bins = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
        
        sr_bin_width = p.BIN_WIDTH
        n_bins_sr = int(round((p.SR_MAX - p.SR_MIN) / sr_bin_width))
        bins_sr = np.linspace(p.SR_MIN, p.SR_MAX, n_bins_sr + 1)
        
        components = ["bkg_Dp", "bkg_Ds", "bkg_Dseta", "bkg_combinatorial"]
        if include_signal:
            components.append("sig")
        
        colors = {
            "bkg_Dp": "C1",
            "bkg_Ds": "C2",
            "bkg_Dseta": "C3",
            "bkg_combinatorial": "C0",
            "sig": "C4"
        }
        labels_legend = {
            "bkg_Dp": "Dp -> 3pi (MisID)",
            "bkg_Ds": "Ds -> 3pi (MisID)",
            "bkg_Dseta": "Ds -> eta mu nu",
            "bkg_combinatorial": "Combinatorial",
            "sig": "Signal"
        }
 
        # =========================================================
        # Plot 1: Full Window vs Signal Region Zoom (Stacked)
        # =========================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        data_full = [x[label == comp] for comp in components]
        color_list = [colors[comp] for comp in components]
        label_list = [labels_legend[comp] for comp in components]
        
        # Full Window panel
        ax1.hist(data_full, bins=bins, stacked=True, color=color_list, label=label_list, edgecolor='black', linewidth=0.3, alpha=0.85)
        ax1.axvspan(p.SR_MIN, p.SR_MAX, color='red', alpha=0.15, label='Signal Region')
        ax1.set_xlabel(r"$m_{3\mu}$ [MeV]", fontsize=12)
        ax1.set_ylabel(f"Events / {bin_width} MeV", fontsize=12)
        ax1.set_title("Full Mass Window", fontsize=14)
        ax1.legend(loc='upper right', fontsize=9)
        
        # SR Zoom panel
        data_sr = [x[(label == comp) & in_sr] for comp in components]
        ax2.hist(data_sr, bins=bins_sr, stacked=True, color=color_list, label=label_list, edgecolor='black', linewidth=0.3, alpha=0.85)
        ax2.set_xlabel(r"$m_{3\mu}$ [MeV]", fontsize=12)
        ax2.set_ylabel(f"Events / {sr_bin_width} MeV", fontsize=12)
        ax2.set_title(f"Signal Region Zoom (N_SR = {in_sr.sum()})", fontsize=14)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "toy_total_breakdown.pdf"), dpi=150, bbox_inches="tight")
        plt.close()
 
        # =========================================================
        # Plot 2 & 3: 4x4 Subplots for the 16 categories (Full & SR)
        # =========================================================
        for mode, win_bins, win_range, title_suffix, filename in [
            ("full", bins, (p.MASS_MIN, p.MASS_MAX), "Full Window", "toy_per_category_full.pdf"),
            ("sr", bins_sr, (p.SR_MIN, p.SR_MAX), "Signal Region", "toy_per_category_sr.pdf")
        ]:
            fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True, sharey=False)
            axes = axes.flatten()
            
            for c in range(p.CATEGORIES):
                ax = axes[c]
                mask_cat = (cat == plot_year * p.CATEGORIES + c)
                if mode == "full":
                    cat_data = [x[mask_cat & (label == comp)] for comp in components]
                else:
                    cat_data = [x[mask_cat & (label == comp) & in_sr] for comp in components]
                
                ax.hist(cat_data, bins=win_bins, stacked=True, color=color_list, edgecolor='none', alpha=0.85)
                if mode == "full":
                    ax.axvspan(p.SR_MIN, p.SR_MAX, color='red', alpha=0.15)
                ax.set_title(f"Category {c}", fontsize=10)
            
            for ax in axes[-4:]:
                ax.set_xlabel(r"$m_{3\mu}$ [MeV]", fontsize=10)
            for ax in axes[::4]:
                ax.set_ylabel("Events", fontsize=10)
                
            fig.suptitle(f"Toy Breakdown per Category ({title_suffix}, {p.YEARS[plot_year]})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
            plt.close()
 
    return x, cat, label
 
if __name__ == "__main__":
    generate_toy(seed=p.SEED, POI=p.POI, include_signal=False, plot=True, verbose=True)
    
    
 
