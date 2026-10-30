import numpy as np
from scipy.stats import poisson, norm
import pyhf


#Va bene per single_bin
def q_tilde(n, b, s, mu):
    """
    Statistica q~_mu per singolo bin senza nuisance.

    n  : numero di eventi osservati, scalare o array
    b  : eventi di background attesi
    s  : eventi di segnale attesi
    mu : parametro di interesse
    """

    n = np.asarray(n, dtype=float)

    if b <= 0:
        raise ValueError("b must be > 0")
    if s <= 0:
        raise ValueError("s must be > 0")
    if mu < 0:
        raise ValueError("mu must be >= 0")

    mu_hat = (n - b) / s

    q = np.zeros_like(n, dtype=float)

    # mu_hat < 0
    mask1 = mu_hat < 0

    q[mask1] = 2.0 * (
        mu * s
        - n[mask1] * np.log1p(mu * s / b)
    )

    # 0 <= mu_hat <= mu
    mask2 = (mu_hat >= 0) & (mu_hat <= mu)

    mask2_npositive = mask2 & (n > 0) 
    mask2_nzero = mask2 & (n == 0) #Se n = 0, python mi da divergenza

    # n > 0
    q[mask2_npositive] = 2.0 * (
        (b + mu * s - n[mask2_npositive])
        - n[mask2_npositive]
        * np.log(
            (b + mu * s) / n[mask2_npositive]
        )
    )

    # n = 0
    q[mask2_nzero] = 2.0 * (b + mu * s)

    # mu_hat > mu -> q = 0

    return q
  
#TOYS
def CLs_toys(n_obs, b, s, mu, N_TOYS = 100_000, rng = None, PRINT = False):
    
    '''
    Calcolo del CLs con il metodo toys
    
    n_obs  : numero di eventi osservati
    b  	   : eventi di background attesi
    s      : eventi di segnale attesi
    mu     : parametro di interesse
    N_TOYS : Numero di toys da generare
    PRINT  : se TRUE, stampa i risultati
    
    '''
    
    if rng is None:
        rng = np.random.default_rng()
    
    toy_b    = rng.poisson(b, size = N_TOYS)
    q_b      = q_tilde(toy_b, b, s, mu)
    
    toy_sb   = rng.poisson(b + mu*s, size = N_TOYS)
    q_sb     = q_tilde(toy_sb, b, s, mu)
    
    ' Calcolo di CLs_obs '
    q_obs    = q_tilde(n_obs, b, s, mu)
    
    CL_b     = np.mean(q_b  >= q_obs)
    CL_sb    = np.mean(q_sb >= q_obs)
    
    if CL_b == 0:
        CLs_obs = np.nan
    else:
        CLs_obs = CL_sb / CL_b
    
    ' Calcolo di CLs_exp '
    
    #Ordino q_s e q_sb per usare searchsorted: O(N^2) vs O(NlogN)
    q_b_sorted = np.sort(q_b) 
    q_sb_sorted = np.sort(q_sb)
        
    q_mu = q_tilde(toy_b, b, s, mu)
        
    idx_b = np.searchsorted(q_b_sorted, q_mu, side="left")
    idx_sb = np.searchsorted(q_sb_sorted, q_mu, side="left")

    CL_b = (N_TOYS - idx_b) / N_TOYS
    CL_sb = (N_TOYS - idx_sb) / N_TOYS
        
    #Escludo CL_b = 0
    CLs_values = np.divide(
    CL_sb,
    CL_b,
    out=np.full_like(CL_sb, np.nan, dtype=float),
    where=CL_b > 0
    )
    
    #Per diagnositca su coda e CL_b = 0
    if PRINT:
        n_zero = np.count_nonzero(CL_b == 0)
        n_small = np.count_nonzero(CL_b < 10 / N_TOYS)

        print("\n--- Diagnostica toys ---")
        print(f"Toy con CL_b = 0              : {n_zero}/{N_TOYS}")
        print(f"Toy con meno di 10 eventi in coda background: "
              f"{n_small}/{N_TOYS}")
        
    
    CLs_exp_median = np.median(CLs_values)
    CLs_exp_minus1 = np.percentile(CLs_values, 16)   # -1σ
    CLs_exp_plus1  = np.percentile(CLs_values, 84)   # +1σ
    CLs_exp_minus2 = np.percentile(CLs_values, 2.5)  # -2σ
    CLs_exp_plus2  = np.percentile(CLs_values, 97.5) # +2σ
    
    CLs_exp = np.array([
        CLs_exp_minus2,
        CLs_exp_minus1,
        CLs_exp_median,
        CLs_exp_plus1,
        CLs_exp_plus2
    ])
    
    if PRINT:
        print("\n=== CALCOLO CLs CON TOYS ===")
        print(f"Observed CLs         = {CLs_obs:.6g}")
        print(f"Expected CLs -2sigma = {CLs_exp_minus2:.6g}")
        print(f"Expected CLs -1sigma = {CLs_exp_minus1:.6g}")
        print(f"Expected CLs median  = {CLs_exp_median:.6g}")
        print(f"Expected CLs +1sigma = {CLs_exp_plus1:.6g}")
        print(f"Expected CLs +2sigma = {CLs_exp_plus2:.6g}")
    
    return CLs_obs, CLs_exp
              

#POISSONIANA
def CLs_poisson(n_obs, b, s, mu, PRINT = False, NMAX = None):
    
    '''
    Calcolo del CLs con il metodo poisson
    
    n_obs  : numero di eventi osservati
    b  	   : eventi di background attesi
    s      : eventi di segnale attesi
    mu     : parametro di interesse
    N_MAX  : numero eventi max per generare funzione poissoniana
    PRINT  : se TRUE, stampa i risultati
    
    '''
    
    #Genero Poissoniana
    if NMAX is None:
        NMAX = poisson.ppf(1 - 1e-12, b + mu*s)
        NMAX = int(NMAX)

    
    n_val 	  = np.arange(0, NMAX + 1)
    
    poisson_b  = poisson.pmf(n_val, b)
    poisson_sb = poisson.pmf(n_val, b + mu * s)
    
    q_mu 	   = q_tilde(n_val, b, s, mu)

    ' Calcolo di CLs_obs '
    q_obs      = q_tilde(n_obs, b, s, mu)
    
    CL_b       = poisson_b [(q_mu >= q_obs)].sum()
    CL_sb	   = poisson_sb[(q_mu >= q_obs)].sum()
    
    if CL_b == 0:
        CLs_obs = np.nan
    else:
        CLs_obs = CL_sb / CL_b
        
    ' Calcolo di CLs exp '
       
    CLs_values = []

    for n in n_val:
        q_obs = q_tilde(n, b, s, mu)
        mask = q_mu >= q_obs

        CL_b = poisson_b[mask].sum()
        CL_sb = poisson_sb[mask].sum()

        if CL_b > 0:
            CLs_values.append(CL_sb / CL_b)
        else:
            CLs_values.append(np.nan)

    CLs_values = np.array(CLs_values)

    valid = np.isfinite(CLs_values)

    sort_mask = np.argsort(CLs_values[valid])
    CLs_sorted = CLs_values[valid][sort_mask]
    cdf = np.cumsum(poisson_b[valid][sort_mask])
    

    CLs_exp_minus2 = CLs_sorted[np.searchsorted(cdf, 0.025)]
    CLs_exp_minus1 = CLs_sorted[np.searchsorted(cdf, 0.16)]
    CLs_exp_median = CLs_sorted[np.searchsorted(cdf, 0.50)]
    CLs_exp_plus1  = CLs_sorted[np.searchsorted(cdf, 0.84)]
    CLs_exp_plus2  = CLs_sorted[np.searchsorted(cdf, 0.975)]
    
    CLs_exp = np.array([
        CLs_exp_minus2,
        CLs_exp_minus1,
        CLs_exp_median,
        CLs_exp_plus1,
        CLs_exp_plus2
    ])
    
    if PRINT:
        print("\n=== CALCOLO CLs CON POISSON ===")
        print(f"Observed CLs         = {CLs_obs:.6g}")
        print(f"Expected CLs -2sigma = {CLs_exp_minus2:.6g}")
        print(f"Expected CLs -1sigma = {CLs_exp_minus1:.6g}")
        print(f"Expected CLs median  = {CLs_exp_median:.6g}")
        print(f"Expected CLs +1sigma = {CLs_exp_plus1:.6g}")
        print(f"Expected CLs +2sigma = {CLs_exp_plus2:.6g}")
    
    return CLs_obs, CLs_exp

#PYHF
def CLs_pyhf(model, n_obs, mu, calctype = "toys", N_TOYS = None, PRINT = False):
    
    '''
    Calcolo del CLs con pyhf
    
    model  : pyhf.Model
    n_obs  : numero di eventi osservati
    mu     : parametro di iteresse
    N_TOYS : Numero di toys da generare
    PRINT  : se TRUE, stampa i risultati
    
    '''
    
    if N_TOYS is None:
        N_TOYS = 100_000
    
    #Metodo toy
    if calctype == "toys":
        CLs_obs, CLs_exp = pyhf.infer.hypotest(
            mu,
            [n_obs],
            model,
            test_stat = "qtilde",
            calctype  = "toybased",
            ntoys     = N_TOYS,
            return_expected_set = True
        )
        
        CLs_exp_minus2 = CLs_exp[0]
        CLs_exp_minus1 = CLs_exp[1]
        CLs_exp_median = CLs_exp[2]
        CLs_exp_plus1  = CLs_exp[3]
        CLs_exp_plus2  = CLs_exp[4]
        
        if PRINT:
            print("\n=== CALCOLO CLs CON PYHF (TOYS) ===")
            print(f"Observed CLs         = {CLs_obs:.6g}")
            print(f"Expected CLs -2sigma = {CLs_exp_minus2:.6g}")
            print(f"Expected CLs -1sigma = {CLs_exp_minus1:.6g}")
            print(f"Expected CLs median  = {CLs_exp_median:.6g}")
            print(f"Expected CLs +1sigma = {CLs_exp_plus1:.6g}")
            print(f"Expected CLs +2sigma = {CLs_exp_plus2:.6g}")
    
    elif calctype == "asymptotics":
        CLs_obs, CLs_exp = pyhf.infer.hypotest(
            mu,
            [n_obs],
            model,
            test_stat = "qtilde",
            calctype  = "asymptotics",
            return_expected_set = True
        )
        
        CLs_exp_minus2 = CLs_exp[0]
        CLs_exp_minus1 = CLs_exp[1]
        CLs_exp_median = CLs_exp[2]
        CLs_exp_plus1  = CLs_exp[3]
        CLs_exp_plus2  = CLs_exp[4]
        
        if PRINT:
            print("\n=== CALCOLO CLs CON PYHF (ASINTOTICO) ===")
            print(f"Observed CLs         = {CLs_obs:.6g}")
            print(f"Expected CLs -2sigma = {CLs_exp_minus2:.6g}")
            print(f"Expected CLs -1sigma = {CLs_exp_minus1:.6g}")
            print(f"Expected CLs median  = {CLs_exp_median:.6g}")
            print(f"Expected CLs +1sigma = {CLs_exp_plus1:.6g}")
            print(f"Expected CLs +2sigma = {CLs_exp_plus2:.6g}")
              
    return CLs_obs, CLs_exp
    
def CLs_asymptotics(n_obs, b, s, mu, PRINT = False) : 
    
    '''
    Calcolo del CLs in modo asintotico tramite Asimov dataset
    
    n_obs  : numero di eventi osservati
    b  	   : eventi di background attesi
    s      : eventi di segnale attesi
    mu     : parametro di interesse
    PRINT  : se TRUE, stampa i risultati
    
    '''	
    
    #Ipotesi di background only
    n_asimov 	  = b
    q_mu_A   	  = q_tilde(n_asimov, b, s, mu)
    sqrt_q_mu_A   = np.sqrt(q_mu_A)
    
    if sqrt_q_mu_A != 0:
    	sigma_mu = mu / sqrt_q_mu_A
    else:
    	sigma_mu = None
    
    ' Calcolo di CLs_obs '
    
    q_mu_obs 	  = q_tilde(n_obs, b, s, mu)
    sqrt_q_mu_obs = np.sqrt(q_mu_obs)
    
    if sqrt_q_mu_obs <= sqrt_q_mu_A:
        t_obs = sqrt_q_mu_obs - sqrt_q_mu_A
        
    else:
        t_obs = (q_mu_obs - q_mu_A) / (2.0 * sqrt_q_mu_A)
    
    CL_sb_obs = norm.cdf(
        -(t_obs + sqrt_q_mu_A)
    )

    CL_b_obs = norm.cdf(
        -t_obs
    )

    if CL_b_obs > 0:
        CLs_obs = CL_sb_obs / CL_b_obs
    else:
        CLs_obs = np.nan
    
    ' Calcolo di CLs_epx '
    
    n_sigma = np.array([
        2.0,
        1.0,
        0.0,
        -1.0,
        -2.0
    ])
    
    CL_sb_exp = norm.cdf(
        -(n_sigma + sqrt_q_mu_A)
    )

    CL_b_exp = norm.cdf(
        -n_sigma
    )

    CLs_exp = CL_sb_exp / CL_b_exp
    
    CLs_exp_minus2 = CLs_exp[0]
    CLs_exp_minus1 = CLs_exp[1]
    CLs_exp_median = CLs_exp[2]
    CLs_exp_plus1  = CLs_exp[3]
    CLs_exp_plus2  = CLs_exp[4]
        
    if PRINT:
        print("\n=== CALCOLO CLs CON ASINTOTICO ===")
        print(f"Observed CLs         = {CLs_obs:.6g}")
        print(f"Expected CLs -2sigma = {CLs_exp_minus2:.6g}")
        print(f"Expected CLs -1sigma = {CLs_exp_minus1:.6g}")
        print(f"Expected CLs median  = {CLs_exp_median:.6g}")
        print(f"Expected CLs +1sigma = {CLs_exp_plus1:.6g}")
        print(f"Expected CLs +2sigma = {CLs_exp_plus2:.6g}")
    
    return CLs_obs, CLs_exp
	
	
def compute_cls(calculator="toys", **kwargs):

    if calculator == "toys":
        return CLs_toys(**kwargs)

    elif calculator == "poisson":
        return CLs_poisson(**kwargs)

    elif calculator == "pyhf":
        return CLs_pyhf(**kwargs)
    
    elif calculator == "asymptotics":
    	return CLs_asymptotics(**kwargs)

    else:
        raise ValueError(f"Unknown method: {calculator}")
