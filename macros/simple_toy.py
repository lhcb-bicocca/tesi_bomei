import numpy as np
from numpy import random
from analysis_helpers.plotting import plot_hist, plot_hists
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pyhf

N_TOYS = 100_000

#Imposto seed per riproducibilità
seed = 42
pyhf.set_backend("numpy")
np.random.seed(42) 

#Hp: Poisson con lambda = b + mu*s

s = 100 	   #segnale atteso
b = 1000 	   #background atteso
n_obs = 1050    #eventi osservati
mu_test = 1    #Supponiamo che segnale esista
	

def q_tilde(n, b, s, mu):
    #Statistica q̃μ per singolo bin senza nuisance
    
    #Ricavo mu_tilde da stima MLE di Poissoniana 
    mu_tilde = (n-b)/s
    
    if mu_tilde < 0:
    	return 2 * (mu*s - n*np.log(1+ (mu*s) / b))
    	
    elif 0 <= mu_tilde <= mu:
    	return 2 * ( (b + mu*s - n) - n*np.log( (b+mu*s) / n ) )
    
    else:
    	return 0 

## Calcolo con toys ##

#q_tilde background
toy_b = random.poisson(b, size = N_TOYS)
q_b_toy = np.array([q_tilde(n, b, s, mu_test) for n in toy_b])

'''
plot_hist(q_b_toy)
plt.show()
'''
    
#q_tilde_signal
toy_sb = random.poisson(b + mu_test*s, size = N_TOYS)
q_sb_toy = np.array([q_tilde(n, b, s, mu_test) for n in toy_sb])

'''
plot_hist(q_sb_toy)
plt.show()

plt.figure()
plt.hist(q_b_toy, label = "background")
plt.hist(q_sb_toy, label = "signal + background")
plt.legend()
plt.show()
'''

#Statistica osservata
q_obs = q_tilde(n_obs, b, s, mu_test)

#Calcolo p_value e CLs
p_sb_toy = np.mean(q_sb_toy >= q_obs)
p_b_toy = np.mean(q_b_toy >= q_obs)
CLs_toy_qtilde = p_sb_toy / p_b_toy

print(f"CLs (toy con qtilde) = {CLs_toy_qtilde}")

'''++++++++++++++++++++++++++'''

## Calcolo da integrale della Poissoniana ##

n_max = (10 * (b + mu_test * s) + 100) #voglio coda abbastanza lunga
n_val = np.arange(0, n_max + 1)

prob_b = poisson.pmf(n_val, b)
prob_sb = poisson.pmf(n_val, b + mu_test * s)

#calcolo q per i vari n
q_value = np.array([q_tilde(n, b, s, mu_test) for n in n_val])

# calcolo p-value e CLs
p_sb = prob_sb[q_value >= q_obs].sum()
p_b = prob_b[q_value >= q_obs].sum()

CLs_qtilde = p_sb / p_b

print(f"q̃_obs = {q_obs}")
print(f"p_sb = {p_sb}")
print(f"p_b = {p_b}")
print(f"CLs = {CLs_qtilde}")

'''++++++++++++++++++++++++++'''

## Usando Pyhf

# Modello senza incertezze
spec = {
    "channels": [
        {
            "name": "single_channel",
            "samples": [
                {
                    "name": "background",
                    "data": [b],
                    "modifiers": []
                },
                {
                    "name": "signal",
                    "data": [s],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None}
                    ]
                }
            ]
        }
    ]
}

model = pyhf.Model(spec, poi_name = "mu")

np.random.seed(42)
CLs_obs_pyhf_toy, CLs_exp_pyhf_toy = pyhf.infer.hypotest(
    mu_test, 
    [n_obs], 
    model, 
    test_stat="qtilde", 
    calctype="toybased",
    ntoys=100_000,
    return_expected_set=True
)

print(f"      Observed CLs (toy): {CLs_obs_pyhf_toy}")
for expected_value, n_sigma in zip(CLs_exp_pyhf_toy, np.arange(-2, 3)):
    print(f"Expected CLs_toy({n_sigma} σ): {expected_value}")


CLs_obs_pyhf, CLs_exp_pyhf = pyhf.infer.hypotest(
    mu_test, 
    [n_obs], 
    model, 
    test_stat="qtilde", 
    calctype="asymptotics",
    return_expected_set=True
)

print(f"      Observed CLs: {CLs_obs_pyhf}")
for expected_value, n_sigma in zip(CLs_exp_pyhf, np.arange(-2, 3)):
    print(f"Expected CLs({n_sigma} σ): {expected_value}")




