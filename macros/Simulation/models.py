import numpy as np
import pyhf
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import exp_integral, gauss_integral, binning
from utils import get_expected_exp_binned, get_expected_gauss_binned, get_expected_flat_binned, get_expected_ref_binned
import parameters as p
import toy_generator
from toy_generator import generate_toy

pyhf.set_backend("numpy", "minuit")

#Calcolo expected for each models given the toy
def models_expected(x, cat, lo, hi, model = "exp", PLOT = False):

    lo, hi = lo, hi
    binned_data, bins = binning(x, cat, lo=lo, hi=hi)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    expected_bkg =  []
    expected_ref =  []
    expected_flat = []
    expected_sig =  []


    for c in range(p.N_CATEGORIES):
        tau = p.TAU_BKG[c]
        k_bkg = exp_integral(lo, hi, tau) / exp_integral(p.SR_MIN, p.SR_MAX, tau)
        n_bkg_window = p.BKG_SR[c]
        
        if model == "flat":
            #Fondo piatto
            flat_binned = get_expected_flat_binned(
                bins=bins, 
                expected_total=n_bkg_window*k_bkg)
            expected_bkg.append(flat_binned)
            
        else:
            # Fondo exp
            bkg_binned = get_expected_exp_binned(
                bins=bins,
                tau=p.TAU_BKG[c],
                expected_total=n_bkg_window,
                lo=p.SR_MIN,
                hi=p.SR_MAX
            )
            expected_bkg.append(bkg_binned)
        
        if model == "refl":
            #Riflessioni attese per categoria c
            ref_binned = get_expected_ref_binned(
                bins, 
                n_bkg_window=n_bkg_window*k_bkg, 
                lo=lo, 
                hi=hi
            )
            expected_ref.append(ref_binned)
        
        # Segnale atteso binnato per la categoria c
        sig_binned = get_expected_gauss_binned(
            bins=bins,
            mu = 0,
            sigma=p.SIGMA_SIG, 
            expected_total=p.N_SIG_C, 
            lo=p.SR_MIN,
            hi=p.SR_MAX
        )
        expected_sig.append(sig_binned)
        
    if PLOT:
        for c in range(p.N_CATEGORIES):
            plt.figure(figsize=(9, 5))
            
            plt.stairs(binned_data[c], bins, label="Toy Data (Observed)", color="black", linewidth=1.5)
            plt.plot(bin_centers, expected_bkg[c], label=f"Bkg ({model})", color="C0", linestyle="--")
            plt.plot(bin_centers, expected_sig[c], label="Expected Signal", color="red", linestyle="-")
            
            total_model = expected_bkg[c] + expected_sig[c]
            
            if model == "refl":
                plt.plot(bin_centers, expected_ref[c], label="Reflections", color="green", linestyle="-.")
                total_model = total_model + expected_ref[c]

            plt.plot(bin_centers, total_model, label="Total Expected Model", color="purple", linewidth=1.8)
            plt.axvspan(p.SR_MIN, p.SR_MAX, color="red", alpha=0.1, label="Signal Region")

            plt.xlabel(r"$m_{3\mu} - m_\tau$ [MeV]")
            plt.ylabel(f"Events / {p.BIN_WIDTH} MeV")
            plt.title(f"Category {c + 1} - Model: {model}")
            plt.legend(fontsize=8, loc="upper right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return binned_data, bins, expected_bkg, expected_sig, expected_ref

def build_pyhf_workspace(
                         binned_data, 
                         bkg_expected, 
                         sig_expected, 
                         cats = None,
                         ref_expected = None
                         ):
    '''
    Build Pyhf Workspace from observed and expected counts from 
    different values.
    '''
    
    if cats is None:
        cats = list(range(p.N_CATEGORIES))
        
    channels =     []
    observations = []
    
    for c in cats:
        cat_name = f"category_{c+1}"
        
        #Toy data
        observations.append({
            "name": cat_name,
            "data": binned_data[c].tolist()
        }) 
            
        #segnale    
        samples = [
            {   
                "name": "signal",
                "data": sig_expected[c].tolist(),
                "modifiers": [
                    #POI
                    {"name": "mu", "type": "normfactor", "data": None},
                    #ALPHA_UNC
                    
                    {
                        "name": "sig_eff_unc",
                        "type": "normsys",
                        "data": {
                            "lo": float(p.ALPHA_LO),
                            "hi": float(p.ALPHA_HI)
                        }
                    },
                    #SIGMA
                ]
            }
        ]
        
        #Background
        samples.append({
            "name": "background",
            "data": bkg_expected[c].tolist(),
            "modifiers": [
                #NORMALIZZAZIONE
                #TAU
            ]
        })    
        
        #Riflessioni
        if ref_expected is not None and np.sum(ref_expected[c]) > 0:
            samples.append({
                "name": "reflections",
                "data": ref_expected[c].tolist(),
                "modifiers": []
                    #incertezza su mu e sigma riflessioni
            })
        channels.append({
            "name": cat_name,
            "samples": samples
        })
            
    #Costruzione JSON
    spec = {
            "channels": channels,
            "observations": observations,
            "measurements": [   
                {
                    "name": "tau3mu_LHCb_run2",
                    "config": {
                        "poi": "mu",
                        "parameters": [
                        {
                            "name": "mu",
                            "bounds": [[0, 100]]  
                        }
                    ]
                    
                    }
                }
            ],
            "version": "1.0.0"
        }
        
    return pyhf.Workspace(spec)     
                
def model_bkg_flat(obs, bkg_flat, sig_exp, cats=None):
    """Genera il workspace pyhf per fondo piatto."""
    return build_pyhf_workspace(obs, bkg_flat, sig_exp, ref_expected=None, cats=cats)

def model_bkg_exp(obs, bkg_exp, sig_exp, cats=None):
    """Genera il workspace pyhf per fondo esponenziale."""
    return build_pyhf_workspace(obs, bkg_exp, sig_exp, ref_expected=None, cats=cats)

def model_bkg_refl(obs, bkg_exp, sig_exp, refl_exp, cats=None):
    """Genera il workspace pyhf per fondo esponenziale + riflessioni."""
    return build_pyhf_workspace(obs, bkg_exp, sig_exp, ref_expected=refl_exp, cats=cats)
                
'''
if __name__ == "__main__":
    x, cat, label = generate_toy(seed=p.SEED, POI=p.POI,
                                 include_signal=True,
                                 plot=False, verbose=False)
    binned_data, bins, expected_bkg, expected_sig, expected_ref = models_expected (x, cat, lo = p.MASS_MIN, hi = p.MASS_MAX, model = "refl", PLOT = False)
'''

'''
def pyhf_workspace(binned_obs, binned_bkg, binned_sig, binned_ref=None,
                              bkg_unc=None, sig_unc=p.ALPHA_REL,
                              categories=None):
'''
