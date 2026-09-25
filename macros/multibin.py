import numpy as np
import pyhf

'''
Tengo 15 bins, come nel paper

Hp 1: fondo b costante con efficienze diverse
Hp 2: fondo b esponenziale

Per il segnale, lo prendo già da paper LHCb run II con fattore di normalizzazione alpha
'''

''' Ipotesi 1 '''

N_BINS = 15
SEED = 42

#Background


b1_ = np.ones(N_BINS) * 1000 #suppongo B = 10
rng = np.random.default_rng(SEED)
eff_b1 = rng.uniform(0.01, 0.1, size=15) #suppongo efficienza tra 1-10 %
print(eff_b1)

b1 = b1_*eff_b1
print(b1)

#Signal

# single-event sensitivity combinata (Tabella 1)
alpha = np.array([1.42e-9, 1.21e-9, 1.05e-9])
sigma_alpha = np.array([0.16e-9, 0.13e-9, 0.11e-9])
rel_unc_alpha = np.sqrt(np.sum((sigma_alpha/alpha**2)**2)) / np.sum(1/alpha)
alpha_hi = 1 / (1+rel_unc_alpha)
alpha_lo = 1 / (1-rel_unc_alpha)

alpha_frac = sigma_alpha / alpha**2
alpha_eff = 1.0 / np.sum(1.0 / alpha)   # ~4.03e-10
sigma_alpha_eff = alpha_eff**2 * np.sqrt(np.sum(alpha_frac**2))

BR_ref = 1e-8
signal_ref = (BR_ref/alpha_eff) * b1/b1.sum() #eventi totali * frazione di eventi per bin

model = pyhf.Model(
    {
        "channels": [
            {
                "name": "cac_cpid_bins",
                "samples": [
                    {
                        "name": "background",
                        "data": b1,
                        "modifiers": []
                    },
                    {
                        "name": "signal",
                        "data": signal_ref.tolist(),
                        "modifiers": [
                            {
                                "name": "mu", 
                                "type": "normfactor", 
                                "data": None
                            },
                            {
                                "name": "alpha_norm",
                                "type": "normsys",
                                "data": {
                                    "hi": alpha_hi, 
                                    "lo": alpha_lo
                                }
                            }
                        ]
                    }
                ]
            }
        ],
        "parameters": [
            {
                "name": "mu", 
                "inits": [0.0], 
                "bounds": [[0.0, 50.0]]
            }
        ]
    },
    poi_name="mu",
)

#Simulo ora i dati osservati con poissoniana per calcolare upper limit
pars = model.config.suggested_init()
pars[model.config.poi_index] = 0

expected_main = model.expected_data(pars, include_auxdata=False)
obs_main = rng.poisson(expected_main)
aux = model.config.auxdata  

# vettore dati completo
obs_data = np.concatenate([obs_main, aux])

obs, exp = pyhf.infer.intervals.upper_limits.upper_limit(
    obs_data, model, np.linspace(0.1, 5, 100), level=0.10)
print("UL atteso (90% CL):", exp[2] * BR_ref)
print("UL osservato (90% CL):", obs * BR_ref)

