import numpy as np
import pyhf
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import exp_integral, gauss_integral, binning
from utils import get_expected_exp_binned, get_expected_gauss_binned, get_expected_flat_binned, get_expected_ref_binned
import parameters as p
import toy_generator
from toy_generator import generate_toy
from stats import fit_background_sidebands

pyhf.set_backend("numpy", "minuit")

#Cache, in this way I don't have to modify analysis.py
_CACHE_BKG_UP = {}
_CACHE_BKG_DOWN = {}
_CACHE_SIG_UP = {}
_CACHE_SIG_DOWN = {}
_CACHE_REFL_UP = {}
_CACHE_REFL_DOWN = {}

#Calcolo expected for each models given the toy
def models_expected(x, cat, lo, hi, model = "exp", PLOT = True, PRINT = False):

    lo, hi = lo, hi
    binned_data, bins = binning(x, cat, lo=lo, hi=hi)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    expected_bkg =  []
    expected_ref =  []
    expected_flat = []
    expected_sig =  []
    
    fit_results_summary = []


    for c in range(p.N_CATEGORIES):
        x_cat = x[cat == c]
        tau = p.TAU_BKG[c]
        k_bkg = exp_integral(lo, hi, tau) / exp_integral(p.SR_MIN, p.SR_MAX, tau)
        n_bkg_window = p.BKG_SR[c]
        
        if model == "flat":
            #Fondo piatto
            flat_binned = get_expected_flat_binned(
                bins=bins, 
                expected_total=n_bkg_window*k_bkg)
            expected_bkg.append(flat_binned)
            
            _CACHE_BKG_UP[(model, c)] = flat_binned
            _CACHE_BKG_DOWN[(model, c)] = flat_binned
            
        elif model == "exp":
            tau_fit, tau_err_down, tau_err_up = fit_background_sidebands(
                x_cat, lo, hi, init_tau=p.TAU_BKG[c], debug=True
            )
            fit_results_summary.append((c + 1, tau_fit, tau_err_down, tau_err_up))
            
            k_bkg_fit = exp_integral(lo, hi, tau_fit) / exp_integral(p.SR_MIN, p.SR_MAX, tau_fit)
            total_bkg_full = n_bkg_window * k_bkg_fit
            
            bkg_binned = get_expected_exp_binned(bins=bins, tau=tau_fit, expected_total=total_bkg_full, lo=lo, hi=hi)
            expected_bkg.append(bkg_binned)
            
            tau_up = tau_fit + tau_err_up
            k_bkg_up = exp_integral(lo, hi, tau_up) / exp_integral(p.SR_MIN, p.SR_MAX, tau_up)
            _CACHE_BKG_UP[(model, c)] = get_expected_exp_binned(bins=bins, tau=tau_up, expected_total=n_bkg_window * k_bkg_up, lo=lo, hi=hi)
            
            tau_down = max(1.0, tau_fit + tau_err_down)
            k_bkg_down = exp_integral(lo, hi, tau_down) / exp_integral(p.SR_MIN, p.SR_MAX, tau_down)
            _CACHE_BKG_DOWN[(model, c)] = get_expected_exp_binned(bins=bins, tau=tau_down, expected_total=n_bkg_window * k_bkg_down, lo=lo, hi=hi)
        
        elif model == "refl":
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
        
        sigma_up = p.SIGMA_SIG * 1.15
        sigma_down = max(0.1, p.SIGMA_SIG * 0.85)
        
        sig_binned_up = get_expected_gauss_binned(
            bins=bins, mu=0, sigma=sigma_up, expected_total=p.N_SIG_C, lo=p.SR_MIN, hi=p.SR_MAX
        )
        sig_binned_down = get_expected_gauss_binned(
            bins=bins, mu=0, sigma=sigma_down, expected_total=p.N_SIG_C, lo=p.SR_MIN, hi=p.SR_MAX
        )
        _CACHE_SIG_UP[c] = sig_binned_up
        _CACHE_SIG_DOWN[c] = sig_binned_down
        
    if model != "flat" and fit_results_summary and PRINT:
        print("\n" + "=" * 65)
        print(f"  RISULTATI FIT PARAMETRO TAU (Modello: {model.upper()})")
        print("=" * 65)
        print(f" {'Categoria'} | {'Tau Fittato':<12} | {'Err Down (Minos)':<16} | {'Err Up (Minos)':<15}")
        print("-" * 65)
        for cat_id, t_val, t_dw, t_up in fit_results_summary:
            print(f" Cat {cat_id:2d}    | {t_val:12.4f} | {t_dw:16.4f} | {t_up:15.4f}")
        print("=" * 65 + "\n")
        
    
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
                         model,
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
            
        down_sig = _CACHE_SIG_DOWN.get(c, sig_expected[c])
        up_sig = _CACHE_SIG_UP.get(c, sig_expected[c])
        
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
                    {
                        "name": f"sig_shape_sigma_cat{c+1}",
                        "type": "histosys",
                        "data": {
                            "lo_data": down_sig.tolist(),
                            "hi_data": up_sig.tolist()
                        }
                    }
                ]
            }
        ]
        
        #Background
        bkg_modifiers = []
        # Cerca i template di bkg nella cache
        down_bkg = _CACHE_BKG_DOWN.get((list(_CACHE_BKG_UP.keys())[0][0] if _CACHE_BKG_UP else "exp", c))
        up_bkg = _CACHE_BKG_UP.get((list(_CACHE_BKG_UP.keys())[0][0] if _CACHE_BKG_UP else "exp", c))
        
        if model == "flat":
            rel_err = float(p.SIGMA_BKG_SR[c] / p.BKG_SR[c])
            bkg_modifiers.append({
                "name": f"bkg_flat_norm_cat{c+1}",
                "type": "normsys",
                "data": {
                    "lo": max(0.0, 1.0 - rel_err),
                    "hi": 1.0 + rel_err
                }
            })
        
        elif model in ["exp", "refl"]:
            bkg_modifiers.append({
                "name": f"bkg_shape_unc_cat{c+1}",
                "type": "histosys",
                "data": {"lo_data": down_bkg.tolist(), "hi_data": up_bkg.tolist()}
            })    
        
        samples.append({
            "name": "background",
            "data": bkg_expected[c].tolist(),
            "modifiers": [] #bkg_modifiers  
        })    
        
        #Riflessioni
        if model == "refl":
            ref_modifiers = [
                {#30% su normalizzazione
                    "name": f"ref_norm_cat{c+1}",
                    "type": "normsys",
                    "data": {"lo": 0.7, "hi": 1.3}
                }
            ]
            samples.append({
                "name": "reflections",
                "data": ref_expected[c].tolist(),
                "modifiers": ref_modifiers
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
                
def model_bkg_flat(obs, bkg_flat, sig_exp, model = "flat", cats=None):
    """Genera il workspace pyhf per fondo piatto."""
    return build_pyhf_workspace(obs, bkg_flat, sig_exp, model, ref_expected=None, cats=cats)

def model_bkg_exp(obs, bkg_exp, sig_exp, model = "exp", cats=None):
    """Genera il workspace pyhf per fondo esponenziale."""
    return build_pyhf_workspace(obs, bkg_exp, sig_exp, model, ref_expected=None, cats=cats)

def model_bkg_refl(obs, bkg_exp, sig_exp, refl_exp, model = "refl", cats=None):
    """Genera il workspace pyhf per fondo esponenziale + riflessioni."""
    return build_pyhf_workspace(obs, bkg_exp, sig_exp, model, ref_expected=refl_exp, cats=cats)
              
def plot_total_fit(x, cat, lo, hi, model="refl"):
    """
    Somma i dati e i modelli attesi di tutte le 15 categorie in un unico istogramma
    e produce il plot globale di controllo.
    """
    # Ottiene i dati binnati e i bin standard
    binned_data, bins = binning(x, cat, lo=lo, hi=hi)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Eseguiamo models_expected per calcolare i vettori per categoria (sfruttando la cache)
    # Nota: questo popolerà anche la cache dei modelli per ciascuna categoria
    from models import models_expected, _CACHE_BKG_UP, _CACHE_BKG_DOWN, _CACHE_REFL_UP, _CACHE_REFL_DOWN
    
    b_data, _, exp_bkg, exp_sig, exp_ref = models_expected(x, cat, lo, hi, model=model, PLOT=False)

    # Somma cumulativa su tutte le categorie
    total_data = np.sum(b_data, axis=0)
    total_bkg = np.sum(exp_bkg, axis=0)
    total_sig = np.sum(exp_sig, axis=0)
    total_ref = np.sum(exp_ref, axis=0) if exp_ref else np.zeros_like(total_bkg)

    total_model = total_bkg + total_sig + total_ref

    # Creazione del plot
    plt.figure(figsize=(10, 6))
    
    plt.stairs(total_data, bins, label="Total Toy Data (Observed)", color="black", linewidth=1.5)
    plt.plot(bin_centers, total_bkg, label=f"Total Background ({model})", color="C0", linestyle="--")
    plt.plot(bin_centers, total_sig, label="Total Expected Signal", color="red", linestyle="-")
    
    if model == "refl" and np.sum(total_ref) > 0:
        plt.plot(bin_centers, total_ref, label="Total Reflections", color="green", linestyle="-.")

    plt.plot(bin_centers, total_model, label="Total Expected Model", color="purple", linewidth=1.8)
    plt.axvspan(p.SR_MIN, p.SR_MAX, color="red", alpha=0.1, label="Signal Region")

    plt.xlabel(r"$m_{3\mu} - m_\tau$ [MeV]", fontsize=12)
    plt.ylabel(f"Events / {p.BIN_WIDTH} MeV", fontsize=12)
    plt.title(f"Aggregated Total Fit (All {p.N_CATEGORIES} Categories Combined) - Model: {model.upper()}")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
                
if __name__ == "__main__":
    x, cat, label = generate_toy(seed=p.SEED, POI=p.POI,
                                 include_signal=True,
                                 plot=False, verbose=False)
    binned_data, bins, expected_bkg, expected_sig, expected_ref = models_expected (x, cat, lo = p.MASS_MIN, hi = p.MASS_MAX, model = "flat", PLOT = False)

    plot_total_fit(x, cat, p.MASS_MIN, p.MASS_MAX, model="flat")
    
    binned_data, bins, expected_bkg, expected_sig, expected_ref = models_expected (x, cat, lo = p.MASS_MIN, hi = p.MASS_MAX, model = "exp", PLOT = False)

    plot_total_fit(x, cat, p.MASS_MIN, p.MASS_MAX, model="exp")
    
    binned_data, bins, expected_bkg, expected_sig, expected_ref = models_expected (x, cat, lo = p.MASS_MIN, hi = p.MASS_MAX, model = "refl", PLOT = False)

    plot_total_fit(x, cat, p.MASS_MIN, p.MASS_MAX, model="refl")
