import numpy as np
import pyhf
import matplotlib.pyplot as plt
from scipy.stats import norm
import parameters as p
from utils import exp_integral, gauss_integral

pyhf.set_backend("numpy", "minuit")

def sample_comb_bkg(rng, n, tau, lo=p.SR_MIN, hi=p.SR_MAX):
	#Sampling n values from exp(-x/tau)
	
    if n <= 0:
        return np.empty(0, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")
    if hi <= lo:
        raise ValueError("hi must be > lo")
        
    a = np.exp(-lo/tau) #pdf values in lo
    b = np.exp(-hi/tau) #pdf values in hi 
    u = rng.random(n)   #n uniform in [0, 1)
    y = a - u*(a - b)   #y in [b, a]
    
    #return x in [lo, hi]
    
    return -tau * np.log(y)
	
def sample_sig(rng, n, mean=0.0, sigma=p.SIGMA_SIG, lo=p.SR_MIN, hi=p.SR_MAX):
	
	#Sampling n values from a gaussian N(mu, sigma)
	
    if n <= 0:
        return np.empty(0, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if hi <= lo:
        raise ValueError("hi must be > lo")
        
    cdf_lo = norm.cdf((lo - mean) / sigma)
    cdf_hi = norm.cdf((hi - mean) / sigma)
    u = rng.uniform(cdf_lo, cdf_hi, size = n)
    
    return mean + sigma*norm.ppf(u)

def generate_toy(
    seed = p.SEED, 
    POI = p.POI, 
    lo = p.MASS_MIN,
    hi = p.MASS_MAX,
    include_signal = True, 
    reflection = True,
    plot=False,
    verbose = True
):

    '''
    Genera un toy event-level.
    Ritorna (x, cat, label, params) dove:
        x     : array di masse
        cat   : array di categorie (0..14)
        label : array di "bkg" o "sig"
    '''
    
    rng = np.random.default_rng(seed)
    all_x, all_cat, all_label = [], [], []

    for c in range (p.N_CATEGORIES):
    
        tau = p.TAU_BKG[c]
        k_bkg = exp_integral(lo, hi, tau) / exp_integral(p.SR_MIN, p.SR_MAX, tau)
        k_sig = (
    norm.cdf(hi / p.SIGMA_SIG) - norm.cdf(lo / p.SIGMA_SIG)
) / (
    norm.cdf(p.SR_MAX / p.SIGMA_SIG) - norm.cdf(p.SR_MIN / p.SIGMA_SIG)
)
	    #signal
        if include_signal:
	        n_sig = rng.poisson(POI * p.N_SIG_C * k_sig)
	        sig   = sample_sig(rng, n_sig, lo=lo, hi=hi)
	        all_x.append(sig)
	        all_cat.append(np.full(n_sig, c))
	        all_label.append(np.full(n_sig, "sig"))
	    
	    #combinatory background
        n_bkg	 = rng.poisson(p.BKG_SR[c] * k_bkg)
        comb_bkg = sample_comb_bkg(rng, n_bkg, tau, lo=lo, hi=hi)
        all_x.append(comb_bkg)
        all_cat.append(np.full(n_bkg, c))
        all_label.append(np.full(n_bkg, "bkg"))
	    
        #reflections
        if reflection:
            N_window = p.BKG_SR[c] * k_bkg
            G = lambda a, b, mu, sigma: (
                norm.cdf((b - mu) / sigma) - norm.cdf((a - mu) / sigma)
            )
            for R in p.REFLECTIONS:
                n_refl = rng.poisson(
                    R["frac"] * N_window
                    * G(lo, hi, R["mu"], R["sigma"])
                    / G(p.MASS_MIN, p.MASS_MAX, R["mu"], R["sigma"])
                )
                refl = sample_sig(rng, n_refl,
                                  mean=R["mu"], sigma=R["sigma"],
                                  lo=lo, hi=hi)
                all_x.append(refl)
                all_cat.append(np.full(n_refl, c))
                all_label.append(np.full(n_refl, "refl"))
                	
    x     = np.concatenate(all_x) if all_x else np.empty(0)
    cat   = np.concatenate(all_cat) if all_cat else np.empty(0, dtype=int)
    label = np.concatenate(all_label) if all_label else np.empty(0, dtype="U4")
    
    #check
    if verbose:
        n_bkg_tot  = (label == "bkg").sum()
        n_refl_tot = (label == "refl").sum()
        n_sig_tot  = (label == "sig").sum()
        in_sr_bkg  = ((x >= p.SR_MIN) & (x <= p.SR_MAX) & (label == "bkg")).sum()
        in_sr_refl = ((x >= p.SR_MIN) & (x <= p.SR_MAX) & (label == "refl")).sum()

        print(f"  Background   : {n_bkg_tot}")
        print(f"  Reflections  : {n_refl_tot}")
        print(f"  Bkg in SR    : {in_sr_bkg}  (atteso ~{p.BKG_SR_TOT:.0f})")
        print(f"  Refl in SR   : {in_sr_refl}")
        if include_signal:
            print(f"  Signal       : {n_sig_tot}")

    if plot:
        bin_width = 10
        n_bins = int(2 * p.MASS_BAND / bin_width)
        bins = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)

        sr_bin_width = 4
        n_bins_sr = int(2 * p.SR_BAND / sr_bin_width)
        bins_sr = np.linspace(p.SR_MIN, p.SR_MAX, n_bins_sr + 1)

        # =========================================================
        # Plot 1: finestra completa + zoom SR
        # =========================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Pannello 1: finestra completa ---
        ax1.hist(x[label == "bkg"], bins=bins, histtype="stepfilled",
                 color="C0", alpha=0.5, edgecolor="navy",
                 label=f"background (N={(label == 'bkg').sum()})")
        if include_signal and (label == "sig").sum() > 0:
            ax1.hist(x[label == "sig"], bins=bins, histtype="stepfilled",
                     color="C3", alpha=0.8, edgecolor="darkred",
                     label=f"signal (N={(label == 'sig').sum()})")
        ax1.axvspan(p.SR_MIN, p.SR_MAX, alpha=0.1, color="red",
                    label="signal region")
        ax1.set_xlabel(r"$m_{3\mu} - m_\tau$ [MeV]")
        ax1.set_ylabel(f"Events / {bin_width} MeV")
        ax1.set_title("Full window")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # --- Pannello 2: zoom SR ---
        ax2.hist(x[label == "bkg"], bins=bins_sr, histtype="stepfilled",
                 color="C0", alpha=0.5, edgecolor="navy", label="background")
        if include_signal and (label == "sig").sum() > 0:
            ax2.hist(x[label == "sig"], bins=bins_sr, histtype="stepfilled",
                     color="C3", alpha=0.8, edgecolor="darkred", label="signal")
        ax2.set_xlabel(r"$m_{3\mu} - m_\tau$ [MeV]")
        ax2.set_ylabel(f"Events / {sr_bin_width} MeV")
        ax2.set_title(f"Signal region zoom (N_SR = {in_sr_bkg + in_sr_refl})")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("toy_total.pdf", dpi=150, bbox_inches="tight")
        plt.show()

        # =========================================================
        # Plot 2: per categoria
        # =========================================================
        fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True)
        axes = axes.flatten()

        for c in range(p.N_CATEGORIES):
            mask_cat = cat == c
            x_bkg = x[mask_cat & (label == "bkg")]
            x_sig = x[mask_cat & (label == "sig")]

            ax = axes[c]
            ax.hist(x_bkg, bins=bins, histtype="stepfilled",
                    color="C0", alpha=0.5, edgecolor="navy",
                    label=f"bkg (N={x_bkg.size})")
            if include_signal and x_sig.size > 0:
                ax.hist(x_sig, bins=bins, histtype="stepfilled",
                        color="C3", alpha=0.8, edgecolor="darkred",
                        label=f"sig (N={x_sig.size})")
            ax.axvspan(p.SR_MIN, p.SR_MAX, alpha=0.1, color="red")
            ax.set_title(f"cat {c + 1}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)

        for ax in axes[-5:]:
            ax.set_xlabel(r"$m_{3\mu} - m_\tau$ [MeV]")
        for ax in axes[::5]:
            ax.set_ylabel(f"Events / {sr_bin_width} MeV")

        fig.suptitle("Toy per category (full window)", fontsize=12)
        plt.tight_layout()
        plt.savefig("toy_per_category_full.pdf", dpi=150, bbox_inches="tight")
        plt.show()
        
        # =========================================================
        # Plot 3: per categoria, zoom sulla signal region
        # =========================================================
        fig, axes = plt.subplots(3, 5, figsize=(16, 9),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        for c in range(p.N_CATEGORIES):
            mask_cat = cat == c
            x_bkg = x[mask_cat & (label == "bkg")]
            x_sig = x[mask_cat & (label == "sig")]

            # Considera solo gli eventi dentro la SR
            x_bkg_sr = x_bkg[(x_bkg >= p.SR_MIN) & (x_bkg <= p.SR_MAX)]
            x_sig_sr = x_sig[(x_sig >= p.SR_MIN) & (x_sig <= p.SR_MAX)]

            ax = axes[c]
            ax.hist(x_bkg_sr, bins=bins_sr, histtype="stepfilled",
                    color="C0", alpha=0.5, edgecolor="navy",
                    label=f"bkg (N={x_bkg_sr.size})")
            if include_signal and x_sig_sr.size > 0:
                ax.hist(x_sig_sr, bins=bins_sr, histtype="stepfilled",
                        color="C3", alpha=0.8, edgecolor="darkred",
                        label=f"sig (N={x_sig_sr.size})")
            ax.set_title(f"cat {c + 1}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)

        for ax in axes[-5:]:
            ax.set_xlabel(r"$m_{3\mu} - m_\tau$ [MeV]")
        for ax in axes[::5]:
            ax.set_ylabel("Events / 1 MeV")

        fig.suptitle("Toy per category (signal region zoom)", fontsize=12)
        plt.tight_layout()
        plt.savefig("toy_per_category_sr.pdf", dpi=150, bbox_inches="tight")
        plt.show()
    
    return x, cat, label

if __name__ == "__main__":
    x, cat, label = generate_toy(seed=p.SEED, POI=p.POI,
                                 include_signal=True,
                                 plot=True, verbose=True)
