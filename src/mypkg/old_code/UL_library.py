from mypkg.CLs_library import compute_cls
from scipy.optimize import toms748
import numpy as np
import pyhf

def upper_limit(n_obs, b, s, alpha = 0.05, 
                calculator = "asymptotics",  model = None, 
                calctype = "asymptotics", PRINT = False, **kwargs):
    
    '''
    Calcolo Upper Limit (sia expected, sia observed)
    
    n_obs      : numero di eventi osservati
    b  	       : eventi di background attesi
    s          : eventi di segnale attesi
    alpha      : significance level
    calculator : calcolatore CLs
    PRINT      : se TRUE, stampa i risultati
    
    '''
    
    ' Calcolo di UL observed '
    
    if calculator == "pyhf":  

        if calctype == "toybased":
        
            mu_up_obs, mu_up_exp, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
                [n_obs], model, level = alpha, 
                return_results=True, calctype = "toybased", track_progress =
                False, **kwargs)
        
        elif calctype == "asymptotics": 
        
            mu_up_obs, mu_up_exp, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
                [n_obs], model, level = alpha,
                return_results=True, calctype = "asymptotics", **kwargs)
    
    else:
    
        def f_obs(mu): 
            CLs_obs, _ = compute_cls(calculator=calculator, 
                         n_obs=n_obs, b=b, s=s, mu = mu)
                    
            return CLs_obs - alpha
        
        #Definisco intervallo con mu_max che varia
        mu_min = 0
        mu_max = 1
        
        while f_obs(mu_max) > 0:
            mu_max *= 2.0
            
        mu_up_obs = toms748(f_obs, mu_min, mu_max)
        
        ' Calcolo di UL expected '
        
        def f_exp(mu, i):

            _, CLs_exp = compute_cls(
                calculator=calculator,
                n_obs=n_obs,
                b=b,
                s=s,
                mu=mu
            )

            return CLs_exp[i] - alpha

        mu_up_exp = []
        
        for i in range(5):

            while f_exp(mu_max, i) > 0:
                mu_max *= 2.0

            root = toms748(
                lambda mu: f_exp(mu, i),
                mu_min,
                mu_max
            )

            mu_up_exp.append(root)

        mu_up_exp = np.array(mu_up_exp)
    
    if PRINT:
        print(f"\n=== CALCOLO UL CON {calculator.upper()} ===")
        print(f"Observed UL         = {mu_up_obs:.6g}")
        print(f"Expected UL -2sigma = {mu_up_exp[0]:.6g}")
        print(f"Expected UL -1sigma = {mu_up_exp[1]:.6g}")
        print(f"Expected UL median  = {mu_up_exp[2]:.6g}")
        print(f"Expected UL +1sigma = {mu_up_exp[3]:.6g}")
        print(f"Expected UL +2sigma = {mu_up_exp[4]:.6g}")

    return mu_up_obs, mu_up_exp
      

