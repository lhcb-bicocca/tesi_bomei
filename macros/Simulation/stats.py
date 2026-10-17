import parameters as p
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil
import matplotlib.pyplot as plt
from iminuit import Minuit
import parameters as p
from utils import exp_integral

#Fit con Minuit di exp

def fit_background_sidebands(x_toy, lo, hi, init_tau=p.TAU_BKG, debug=False):
    
    sideband_mask = (x_toy < p.SR_MIN) | (x_toy > p.SR_MAX)
    x_sidebands = x_toy[sideband_mask]
    
    if len(x_sidebands) < 5:
        if debug:
            print(f"[DEBUG] Troppo pochi eventi nelle sidebands ({len(x_sidebands)}). Uso init: tau = {init_tau}")
            #Variazione del 20%
        return init_tau, -0.2 * init_tau, 0.2 * init_tau
        
    def nll_exp(tau):
        if tau <= 0:
            return 1e10
        norm_factor = exp_integral(lo, hi, tau)
        if norm_factor <= 0:
            return 1e10
        
        pdf_vals = np.exp(-x_sidebands / tau) / norm_factor
        pdf_vals = np.maximum(pdf_vals, 1e-300)
        return -np.sum(np.log(pdf_vals))

    # 3. Fit con iminuit e blocchi di debug
    try:
        m = Minuit(nll_exp, tau=init_tau)
        
        if debug:
            print("\n" + "=" * 50)
            print(f"[DEBUG] INIZIO FIT SIDEBANDS (Eventi validi: {len(x_sidebands)})")
            print("=" * 50)
            
        # Esecuzione MIGRAD
        m.migrad()
        if debug:
            print(f"[1/3] MIGRAD completato:")
            print(f"  - Convergenza valida (is_valid): {m.valid}")
            print(f"  - Valore minimo della NLL (m.fval): {m.fval:.4f}")
            print(f"  - Stato fmin: {m.fmin}")
            print(f"  - Valore provvisorio tau: {m.values['tau']:.4f}")

        # Esecuzione HESSE
        m.hesse()
        if debug:
            print(f"[2/3] HESSE completato:")
            print(f"  - Errore parabolico (Hesse): {m.errors['tau']:.4f}")

        # Esecuzione MINOS 
        has_minos = False
        try:
            m.minos()
            if 'tau' in m.merrors and m.merrors['tau'].is_valid:
                has_minos = True
                if debug:
                    print(f"[3/3] MINOS completato:")
                    print(f"  - Errori asimmetrici: -{m.merrors['tau'].lower:.4f}, +{m.merrors['tau'].upper:.4f}")
        except Exception:
            if debug:
                print("[3/3] MINOS non è riuscito a calcolare gli errori asimmetrici.")

        if not m.valid or m.errors["tau"] is None or np.isnan(m.errors["tau"]):
            if debug:
                print("[DEBUG] Risultato del fit scartato (valido=False o errore NaN). Applico fallback.")
                print("=" * 50 + "\n")
            return init_tau, -0.2 * init_tau, 0.2 * init_tau
            
        if debug:
            print(f"[DEBUG] FIT RIUSCITO CON SUCCESSO: tau = {m.values['tau']:.4f}")
            print("=" * 50 + "\n")
        

        if has_minos:
            return m.values["tau"], m.merrors["tau"].lower, m.merrors["tau"].upper
        else:
            return m.values["tau"], -m.errors["tau"], m.errors["tau"]
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] Eccezione critica durante il fit: {e}")
            print("=" * 50 + "\n")
        return init_tau, -0.2 * init_tau, 0.2 * init_tau

def fit_background_sidebands_with_refl(x_toy, lo, hi, init_tau=p.TAU_BKG, debug=True):
    """
    Fit delle sidebands per il modello esponenziale + riflessioni note.
    """
    sideband_mask = (x_toy < p.SR_MIN) | (x_toy > p.SR_MAX)
    x_sidebands = x_toy[sideband_mask]
    
    if len(x_sidebands) < 5:
        if debug:
            print(f"[DEBUG] Troppo pochi eventi nelle sidebands ({len(x_sidebands)}). Uso init: tau = {init_tau}")
        return init_tau, -0.2 * init_tau, 0.2 * init_tau
        
    # Calcoliamo le normalizzazioni e le forme delle riflessioni sulle sidebands
    # Dalle impostazioni in parameters, le riflessioni sono gaussiane note.
    # Possiamo pre-calcolare le frazioni attese.
    
    def nll_exp_plus_refl(tau):
        if tau <= 0:
            return 1e10
        
        # 1. Integrale della componente esponenziale
        norm_exp = exp_integral(lo, hi, tau)
        if norm_exp <= 0:
            return 1e10
        
        # PDF esponenziale normalizzata
        pdf_exp = np.exp(-x_sidebands / tau) / norm_exp
        
        # 2. Componenti di riflessione (fisse come modellate in p.REFLECTIONS)
        pdf_refl_sum = np.zeros_like(x_sidebands)
        total_refl_frac = sum(R["frac"] for R in p.REFLECTIONS)
        
        for R in p.REFLECTIONS:
            # Integrale della gaussiana della riflessione nella finestra [lo, hi]
            g_int = gauss_integral(lo, hi, R["mu"], R["sigma"])
            if g_int > 0:
                g_vals = norm.pdf(x_sidebands, loc=R["mu"], scale=R["sigma"]) / g_int
                # Pesa la riflessione in base alla sua frazione relativa rispetto al fondo
                pdf_refl_sum += (R["frac"] / (1.0 + total_refl_frac)) * g_vals

        # Frazione di fondo combinatorio puro vs riflessioni
        frac_comb = 1.0 / (1.0 + total_refl_frac)
        frac_refl = total_refl_frac / (1.0 + total_refl_frac) # approssimazione cumulativa o gestita singolarmente
        
        # PDF Totale del fondo nelle sidebands = frac_comb * PDF_exp + frac_refl * PDF_refl
        # (In modo rigoroso, le frazioni dipendono da N_window, ma qui normalizziamo le PDF)
        pdf_total = frac_comb * pdf_exp + pdf_refl_sum
        pdf_total = np.maximum(pdf_total, 1e-300)
        
        return -np.sum(np.log(pdf_total))

    # Esecuzione del fit con Minuit (identica alla tua struttura precedente)
    try:
        m = Minuit(nll_exp_plus_refl, tau=init_tau)
        m.migrad()
        m.hesse()
        
        has_minos = False
        try:
            m.minos()
            if 'tau' in m.merrors and m.merrors['tau'].is_valid:
                has_minos = True
        except Exception:
            pass

        if not m.valid or m.errors["tau"] is None or np.isnan(m.errors["tau"]):
            return init_tau, -0.2 * init_tau, 0.2 * init_tau
            
        if has_minos:
            return m.values["tau"], m.merrors["tau"].lower, m.merrors["tau"].upper
        else:
            return m.values["tau"], -m.errors["tau"], m.errors["tau"]
            
    except Exception as e:
        return init_tau, -0.2 * init_tau, 0.2 * init_tau

#Upper limit con pyhf
def upper_limit(workspace, level = p.LEVEL, PLOT = False, VERBOSE = True):
    model = workspace.model()
    data  = workspace.data(model)

    ul_obs_grid, ul_exp_grid = pyhf.infer.intervals.upper_limits.upper_limit(data, model, np.linspace(0,50,100), level=level)
    
    BR_obs_grid = ul_obs_grid * p.BR_REF
    BR_exp_grid = np.asarray(ul_exp_grid) * p.BR_REF
    
    if VERBOSE:
	    print(f"UPPER LIMIT CON ASINTOTICO E GRID")
	    print(f"BR_obs (90% CL) = {BR_obs_grid}")
	    print("BR expected (90% CL):")
	    print(f"  median         = {BR_exp_grid[2]}")
	    print(f"  -2σ            = {BR_exp_grid[0]}")
	    print(f"  -1σ            = {BR_exp_grid[1]}")
	    print(f"  +1σ            = {BR_exp_grid[3]}")
	    print(f"  +2σ            = {BR_exp_grid[4]}")
	    print("\n")

	#Asymptotic with rootfinder (toms748)
	
    ul_obs_root, ul_exp_root = pyhf.infer.intervals.upper_limits.upper_limit(data, model, level=level)
    
    BR_obs_root = ul_obs_root * p.BR_REF
    BR_exp_root = np.asarray(ul_exp_root) * p.BR_REF

    if VERBOSE:
	    print(f"UPPER LIMIT CON ASINTOTICO E TOMS748")
	    print(f"BR_obs (90% CL) = {BR_obs_root}")
	    print("BR expected (90% CL):")
	    print(f"  median         = {BR_exp_root[2]}")
	    print(f"  -2σ            = {BR_exp_root[0]}")
	    print(f"  -1σ            = {BR_exp_root[1]}")
	    print(f"  +1σ            = {BR_exp_root[3]}")
	    print(f"  +2σ            = {BR_exp_root[4]}")
	    print("\n")
	
	
    if PLOT:
	    #brazil plot
	    test_mus = np.linspace(0, 10, 100)
	    results = [pyhf.infer.hypotest(m, data, model, test_stat="qtilde", return_expected_set=True) for m in test_mus]

	    fig, ax = plt.subplots()
	    brazil.plot_results(test_mus*p.BR_REF, results, test_size = level, ax=ax)

	    ax.set_xlabel("Br", fontsize=12)
	    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	    ax.set_ylim(0, 1.0)

	    plt.show()
	    
    return BR_obs_grid, BR_exp_grid, BR_obs_root, BR_exp_root
