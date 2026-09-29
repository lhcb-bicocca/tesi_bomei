import numpy as np
import pyhf
from pyhf.contrib.viz import brazil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

pyhf.set_backend("numpy", "minuit")

N_BINS           = 15
SEED             = 42
CONFIDENCE_LEVEL = 0.10
BR_ref           = 1e-8
ALPHA            = 4.0273e-10 #See multibin_cost for derivation
MUON_MASS        = 105.6583755 #Mev
TAU_MASS         = 1776.93 #MEV
Resonance        = TAU_MASS
rng_obs          = np.random.default_rng(SEED + 2)

sigma_alpha = np.array([0.16e-9, 0.13e-9, 0.11e-9])
rel_unc_alpha = np.sqrt(np.sum((sigma_alpha/ALPHA**2)**2)) / np.sum(1/ALPHA)
alpha_hi = 1 / (1+rel_unc_alpha)
alpha_lo = 1 / (1-rel_unc_alpha)

edges = np.linspace(-150, 150, N_BINS+1)      # ±150 MeV, 15 bin da 20 MeV
centers = 0.5*(edges[:-1]+edges[1:])


#BACKGROUND
'''
I want b(x) = b0 * exp(-tau*x)
I can suppose same efficiencty? 
Main uncertainty regards parameter tau
Is it correct to take the gamma value from the different processes and make a mean?

'''

B_norm = 1444
sigma_b_norm = 17
B_norm_hi = B_norm + sigma_b_norm
B_norm_lo = B_norm - sigma_b_norm

bkg_tau = 130

def bkg_shape(x, B, tau):
    raw = np.exp(-x / tau)
    return B * raw / raw.sum()

bkg = bkg_shape(centers, B_norm, bkg_tau)

#SIGNAL
'''
I want s(x) ~ gauss, where mean is m(3mu), sigma is my free parameter (technically, I can find a row value  from relativistic BW in the resonance.

It should be something like Ns*exp(-0.5*(x/sigma)²), where Ns is, as before, defined as Br/alpha

'''

S_norm = BR_ref / ALPHA
sigma = 10

def sig_shape(x, N, sigma):
    raw = np.exp(-0.5 * (x / sigma)**2)
    return N * raw / raw.sum()

sig = sig_shape(centers, S_norm, sigma)


plt.figure()
plt.bar(centers, bkg, width=(edges[1]-edges[0]),
        alpha=0.5, label="bkg per bin", edgecolor='black')
plt.bar(centers, sig, width=(edges[1]-edges[0]),
        alpha=0.5, label="sig per bin", edgecolor='yellow') 
plt.show()

#INCERTEZZE SU PARAMETRI PER HISTOSYS
delta_tau   = 0.01 * bkg_tau   # incertezza su tau
delta_sigma = 0.01 * sigma     # incertezza su sigma

bkg_hi      = bkg_shape(centers, B_norm, bkg_tau + delta_tau)
bkg_lo      = bkg_shape(centers, B_norm, bkg_tau - delta_tau)

sig_hi      = sig_shape(centers, S_norm, sigma + delta_sigma)
sig_lo      = sig_shape(centers, S_norm, sigma - delta_sigma)

print("bkg nominal sum:", bkg.sum())
print("bkg hi sum:     ", bkg_hi.sum())
print("bkg lo sum:     ", bkg_lo.sum())
print("sig nominal sum:", sig.sum())
print("sig hi sum:     ", sig_hi.sum())
print("sig lo sum:     ", sig_lo.sum())

#ANALYSIS
'''
Except for JSON structure (e.g. histosys and not shapesys for the parameters), the rest is practically the same
'''


#Definisco il modello JSON
model = pyhf.Model(
    {
        "channels": [
            {
                "name": "cac_cpid_bins",
                "samples": [
                    {
                        "name": "background",
                        "data": bkg,
                        "modifiers": [
                            {
                                "name": "unc_tau",
                                "type": "histosys",
                                "data": {
                                    "hi_data": bkg_hi.tolist(),
                                    "lo_data": bkg_lo.tolist()   
                                },
                            },
                            {
                            	"name": "b_norm",
                            	"type": "normsys",
                            	"data": {
                            		"hi": B_norm_hi,
                            		"lo": B_norm_lo
                            	}
                            },	     
                        ]
                    },
                    {
                        "name": "signal",
                        "data": sig,
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
                                },
                            },
                            {
                                "name": "unc_sigma",
                                "type": "histosys",
                                "data": {
                                    "hi_data": sig_hi.tolist(),
                                    "lo_data": sig_lo.tolist()
                                },
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
                "bounds": [[0.0, 100.0]]
            }
        ]
    },
    poi_name="mu",
)

#Simulo dati osservati
pars = model.config.suggested_init()
pars[model.config.poi_index] = 0
expected_main = model.expected_data(pars, include_auxdata=False)
obs_main = rng_obs.poisson(expected_main)
aux = model.config.auxdata  

#obs_main = np.array([284,174,110,203,121,67,41,133,62,50,42,79,33,29,23])
obs_data = np.concatenate([obs_main, aux])

#Calcolo upper limit con vari metodi per confrontare

#Asymptotic with grid
ul_obs_grid, ul_exp_grid = pyhf.infer.intervals.upper_limits.upper_limit(obs_data, model, np.linspace(0,50,1000), level=CONFIDENCE_LEVEL)

BR_obs_grid = ul_obs_grid * BR_ref
BR_exp_grid = np.asarray(ul_exp_grid) * BR_ref

print("UPPER LIMIT CON ASINTOTICO E GRID")
print(f"BR_obs (90% CL) = {BR_obs_grid}")
print("BR expected (90% CL):")
print(f"  median         = {BR_exp_grid[2]}")
print(f"  -2σ            = {BR_exp_grid[0]}")
print(f"  -1σ            = {BR_exp_grid[1]}")
print(f"  +1σ            = {BR_exp_grid[3]}")
print(f"  +2σ            = {BR_exp_grid[4]}")
print("\n")

#Asymptotic with rootfinder (toms748)
ul_obs_root, ul_exp_root = pyhf.infer.intervals.upper_limits.upper_limit(obs_data, model, level=CONFIDENCE_LEVEL)

BR_obs_root = ul_obs_root * BR_ref
BR_exp_root = np.asarray(ul_exp_root) * BR_ref

print("UPPER LIMIT CON ASINTOTICO E TOMS748")
print(f"BR_obs (90% CL) = {BR_obs_root}")
print("BR expected (90% CL):")
print(f"  median         = {BR_exp_root[2]}")
print(f"  -2σ            = {BR_exp_root[0]}")
print(f"  -1σ            = {BR_exp_root[1]}")
print(f"  +1σ            = {BR_exp_root[3]}")
print(f"  +2σ            = {BR_exp_root[4]}")
print("\n")


'''
#Toy con grid; oss: con ntoys = 200, circa 2 minuit per mu
ul_obs_toy, ul_exp_toy = pyhf.infer.intervals.upper_limits.upper_limit(
    obs_data, model, np.linspace(0,10,100), level=CONFIDENCE_LEVEL,
    calctype="toybased", ntoys=200)   
    
BR_obs_toy = ul_obs_toy * BR_ref
BR_exp_toy = np.asarray(ul_exp_toy) * BR_ref

print("UPPER LIMIT CON TOYS")
print(f"BR_obs (90% CL) = {BR_obs_toy}")
print("BR expected (90% CL):")
print(f"  median         = {BR_exp_toy[2]}")
print(f"  -2σ            = {BR_exp_toy[0]}")
print(f"  -1σ            = {BR_exp_toy[1]}")
print(f"  +1σ            = {BR_exp_toy[3]}")
print(f"  +2σ            = {BR_exp_toy[4]}")
print("\n")
'''


# Verifica convergenza con Minuit
print("Comparison of upper limits")

# --- Valori dei limiti nei due metodi ---
print("\n[Upper limits on mu]")
print(f"{'':12s} {'grid scan':>15s} {'toms748':>15s} {'diff':>12s}")
print("-" * 70)

# Osservato
diff_obs = abs(ul_obs_grid - ul_obs_root)
print(f"{'observed':12s} {ul_obs_grid:>15.6f} {ul_obs_root:>15.6f} {diff_obs:>12.2e}")

# Atteso (mediana)
med_grid = ul_exp_grid[2]
med_root = ul_exp_root[2]
diff_med = abs(med_grid - med_root)
print(f"{'expected':12s} {med_grid:>15.6f} {med_root:>15.6f} {diff_med:>12.2e}")

# p-value al limite: deve essere ~0.10 in entrambi 
print(f"{'':12s} {'grid scan':>15s} {'toms748':>15s}")
print("-" * 70)

pairs = [
    ("observed", ul_obs_grid, ul_obs_root),
    ("expected", med_grid,    med_root),
]

for label, mu_grid, mu_root in pairs:
    pval_grid = float(pyhf.infer.hypotest(
        mu_grid, obs_data, model, test_stat="qtilde"
    ))
    pval_root = float(pyhf.infer.hypotest(
        mu_root, obs_data, model, test_stat="qtilde"
    ))
    print(f"{label:12s} {pval_grid:>15.4f} {pval_root:>15.4f}")

#brazil plot
test_mus = np.linspace(0, 10, 100)
results = [pyhf.infer.hypotest(m, obs_data, model, test_stat="qtilde", return_expected_set=True) for m in test_mus]

fig, ax = plt.subplots()
brazil.plot_results(test_mus*BR_ref, results, test_size = CONFIDENCE_LEVEL, ax=ax)

ax.set_xlabel("Br", fontsize=12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0, 1.0)

plt.show()

