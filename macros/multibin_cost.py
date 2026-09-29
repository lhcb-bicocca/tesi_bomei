import numpy as np
import pyhf
from pyhf.contrib.viz import brazil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

pyhf.set_backend("numpy", "minuit")

'''
~ Multibin with constant background ~
I suppose 15 bins as in the paper of LHCb run II (actually 15 x 3 years, in this simple I approximate with 15 bins and I take the weighted mean for alpha).

For each bin, I assign an efficiency generated with random function, differently for background and signal.

For the signal, I take the reference from LCHB run II paper, where s = BR /alpha, with alpha normalization factor.

'''

N_BINS           = 15
SEED             = 42
CONFIDENCE_LEVEL = 0.10
rng_bkg          = np.random.default_rng(SEED)
rng_sig          = np.random.default_rng(SEED + 1)
rng_obs          = np.random.default_rng(SEED + 2)
BR_ref           = 1e-8


#Background
bkg_exp = np.ones(N_BINS) * 1000 #suppongo 1000 eventi
eff_bkg = rng_bkg.uniform(0.01, 0.1, size=15) #suppongo eff tra 1-10%
bkg     = bkg_exp * eff_bkg

sigma_eff_bkg = eff_bkg*np.sqrt( (1-eff_bkg) / bkg )
bkg_unc       = (bkg* sigma_eff_bkg / eff_bkg ).tolist()

print("_____BACKGROUND_____")
print(f"bkg_exp       : {bkg_exp[0]}")
print(f"eff_bkg       : {eff_bkg}")
print(f"sigma_eff_bkg : {sigma_eff_bkg}")
print(f"bkg           : {bkg}")
print(f"bkg_unc       : {bkg_unc}")
print("\n")

#Signal
alpha       = np.array([1.42e-9, 1.21e-9, 1.05e-9]) #Single-event sensitivity
sigma_alpha = np.array([0.16e-9, 0.13e-9, 0.11e-9])

rel_unc_alpha = np.sqrt(np.sum((sigma_alpha/alpha**2)**2)) / np.sum(1/alpha)
alpha_hi = 1 / (1+rel_unc_alpha)
alpha_lo = 1 / (1-rel_unc_alpha)

alpha_eff       = 1.0 / np.sum(1.0 / alpha)
frac            = (sigma_alpha) / (alpha**2)
sigma_alpha_eff = alpha_eff**2 * np.sqrt(np.sum(frac**2))

eff_sig = rng_sig.uniform(0.2, 0.6, size=N_BINS) #suppongo eff tra 20-60 %
sig     = (BR_ref/alpha_eff) * (eff_sig/eff_sig.sum())

sigma_eff_sig = eff_sig*np.sqrt( (1-eff_sig) / sig )
sig_unc       = (sig* sigma_eff_sig / eff_sig ).tolist()

print("_____SIGNAL_____")
print(f"alpha_eff       : {alpha_eff}")
print(f"sigma_alpha_eff : {sigma_alpha_eff}")
print(f"eff_sig         : {eff_sig}")
print(f"sigma_eff_sig   : {sigma_eff_sig}")
print(f"sig             : {sig}")
print(f"sig_unc         : {sig_unc}")
print("\n")

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
                                "name": "bkg_shape",
                                "type": "shapesys",
                                "data": bkg_unc
                            }
                        ]
                    },
                    {
                        "name": "signal",
                        "data": sig.tolist(),
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
                                "name": "sig_shape",
                                "type": "shapesys",
                                "data": sig_unc
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
ul_obs_grid, ul_exp_grid = pyhf.infer.intervals.upper_limits.upper_limit(obs_data, model, np.linspace(0,10,100), level=CONFIDENCE_LEVEL)

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

#  Differenze relative 
print("\n[Relative differences]")
if ul_obs_root != 0:
    print(f"  observed: {diff_obs / ul_obs_root * 100:.4f}%")
if med_root != 0:
    print(f"  expected: {diff_med / med_root * 100:.4f}%")


#brazil plot
test_mus = np.linspace(0, 6, 100)
results = [pyhf.infer.hypotest(m, obs_data, model, test_stat="qtilde", return_expected_set=True) for m in test_mus]

fig, ax = plt.subplots()
brazil.plot_results(test_mus*BR_ref, results, test_size = CONFIDENCE_LEVEL, ax=ax)

ax.set_xlabel("Br", fontsize=12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0, 1.0)

plt.show()

