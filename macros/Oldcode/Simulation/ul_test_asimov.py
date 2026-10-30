import numpy as np
import pyhf
from pyhf.infer.intervals.upper_limits import upper_limit
from pyhf.infer.calculators import generate_asimov_data
import logging

import parameters as p
from toy_generator import generate_toy
from models import build_workspace

logging.getLogger("pyhf").setLevel(logging.ERROR)
pyhf.set_backend("jax", "scipy")

def main():
    print("Generazione dei dati nominali/toy di base...")
    x, cat, label = generate_toy(
        seed = p.SEED, 
        POI = 0, 
        include_signal = False, 
        MisId = True, 
        Dseta = True, 
        verbose = False
    )
    
    try:
        print("Costruzione del workspace pyhf...")
        ws = build_workspace(
            x = x,
            cat = cat,
            label = "asimov_test",
            model_type = "exp_misid_dseta",
            exclude_cat = [],              # Metti le categorie o l'anno che vuoi testare
            signal_syst = True,
            misid_syst = True,
            use_staterror = True
        )
        
        model = ws.model()
        nominal_data = ws.data(model)
        
        print("Estrazione del dataset Asimov (mu = 0.0)...")
        asimov_data = generate_asimov_data(0.0, nominal_data, model)
        
        print("Esecuzione del calcolo dell'Upper Limit asintotico sul dataset Asimov (95% CL)...")
        obs_limit, exp_limits = upper_limit(
            asimov_data, 
            model, 
            level = 0.05, 
            test_stat = "qtilde"
        )
        
        print("\n================ RISULTATO ASIMOV ASINTOTICO ================")
        print(f"Asimov Upper Limit (95% CL): {float(obs_limit):.4f}")
        print(f"Expected Upper Limit (Median): {float(exp_limits[2]):.4f}")
        print("=============================================================\n")
        print("Test con Asimov completato con successo!")
        
    except Exception as e:
        print(f"\n[ERRORE]: {e}")
        raise e

if __name__ == "__main__":
    main()
