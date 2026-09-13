from mypkg.CLs_library import compute_cls
from scipy.optimize import toms748

def upper_limit(n_obs, b, s, alpha = 0.05, calculator = "asymtotics"):
    
    def f(mu): 
        CLs_obs, _ = compute_cls(calculator=calculator, 
                     n_obs=n_obs, b=b, s=s, mu = mu)
                
        return CLs_obs - alpha
    
    #Definisco intervallo con mu_max che varia
    mu_min = 0
    mu_max = 1
    
    while f(mu_max) > 0:
        mu_max *= 2.0
        
    mu_up = toms748(f, mu_min, mu_max)
    
    return mu_up   
      

