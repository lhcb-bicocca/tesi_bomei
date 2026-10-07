import numpy as np
from scipy.stats import norm
import parameters as p

def exp_integral(lo, hi, tau):
    
    if tau is None:
        tau = p.TAU_BKG    
    
    return tau * (
        np.exp(-lo / tau) -
        np.exp(-hi / tau)
    )
   
def gauss_integral(lo, hi, mu, sigma):     
    
    return ( 
        norm.cdf( (hi - mu) / sigma) -
        norm.cdf( (lo - mu) / sigma)
    )
   
   
def binning(x, cat, lo, hi, n_categories = p.N_CATEGORIES, bin_width = p.BIN_WIDTH):
    '''
    Divido eventi del toy in istogrammi per ciascuna categoria
    return tupla (binned_data, bins)
        - binned_data = array (one for each category) with the counts
        - bins        = edges of calculated bins
    '''
    n_bins = int(np.round((hi - lo) /bin_width))
    bins   = np.linspace(lo, hi, n_bins + 1)
    
    binned_data = []
    
    for c in range(n_categories):
        x_cat = x[cat == c]
        counts, _ = np.histogram(x_cat, bins=bins)
        binned_data.append(counts)
        
    return binned_data, bins

def get_expected_exp_binned(bins, expected_total, lo, hi, tau = p.TAU_BKG):
    """
    Calcola i conteggi analitici attesi per un fondo esponenziale in ogni bin.
    """
    norm_tot = exp_integral(lo, hi, tau)
    
    expected_bins = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        b_lo, b_hi = bins[i], bins[i+1]
        # Frazione di eventi attesi in questo specifico bin
        frac = exp_integral(b_lo, b_hi, tau) / norm_tot
        expected_bins[i] = expected_total * frac
        
    return expected_bins

def get_expected_gauss_binned(bins, mu, sigma, expected_total, lo, hi):
    """
    Calcola i conteggi analitici attesi per un segnale gaussiano in ogni bin.
    """
    # Integrale totale sulla finestra per normalizzare la PDF
    norm_tot = gauss_integral(lo, hi, mu, sigma)
    
    expected_bins = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        b_lo, b_hi = bins[i], bins[i+1]
        frac = gauss_integral(b_lo, b_hi, mu, sigma) / norm_tot
        expected_bins[i] = expected_total * frac
        
    return expected_bins

def get_expected_flat_binned(bins, expected_total):
    """
    Calcola i conteggi attesi per un fondo costante in ogni bin.
    """
    expected_bins = np.zeros(len(bins) - 1)
    total_width = bins[-1] - bins[0]
    
    for i in range(len(bins) - 1):
        bin_width = bins[i+1] - bins[i]
        expected_bins[i] = expected_total * (bin_width / total_width)
        
    return expected_bins

def get_expected_ref_binned(bins, n_bkg_window, lo, hi, reflections=p.REFLECTIONS):
    """
    Calcola la somma binnata di tutte le componenti di riflessione.
    """
    ref_binned_tot = np.zeros(len(bins) - 1)
    for R in reflections:
        G_win = gauss_integral(lo, hi, R["mu"], R["sigma"])
        G_full = gauss_integral(p.MASS_MIN, p.MASS_MAX, R["mu"], R["sigma"])
        
        # Eventi totali attesi per questa specifica riflessione
        n_refl_expected = R["frac"] * n_bkg_window * (G_win / G_full)
        
        ref_binned_tot += get_expected_gauss_binned(
            bins, mu=R["mu"], sigma=R["sigma"],
            expected_total=n_refl_expected, lo=lo, hi=hi
        )
    return ref_binned_tot







