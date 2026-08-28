import numpy as np
from numpy import random
from analysis_helpers.plotting import plot_hist, plot_hists
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pyhf

N_TOYS = 100_000

#Hp: Poisson con lambda = b + mu*s

s = 10 		#segnale atteso
b = 70 		#background atteso
n_obs = 60    #eventi osservati
mu_test = 1 #Supponiamo che segnale esista
	

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

#Calcolo p_value e CLs)
p_sb_toy = np.mean(q_sb_toy >= q_obs)
p_b_toy = np.mean(q_b_toy >= q_obs)
CLs_toy_qtilde = p_sb_toy / (1 - p_b_toy)

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

CLs_qtilde = p_sb / (1 - p_b)

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
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None}
                    ]
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

model = pyhf.Model(spec)

'''
#Non funziona. Probabile conflitto di versione con numpy? Da approfondire
CLs_pyhf_toy_qtilde = pyhf.infer.hypotest(
    mu_test, 
    [n_obs], 
    model, 
    test_stat="qtilde", 
    calctype="toybased",
    ntoys=100_000,
    return_expected=False
)

print(f"CLs (pyhf con qtilde, toys) = {CLs_pyhf_toy_qtilde}")
'''

CLs_obs_pyhf = pyhf.infer.hypotest(
    mu_test, 
    [n_obs], 
    model, 
    test_stat="qtilde", 
    calctype="asymptotics",
    return_expected=False
)

print(f"CLs (pyhf, asintotico): {CLs_obs_pyhf}")


