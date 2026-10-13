import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.contrib.viz import brazil

import parameters as p
from toy_generator import generate_toy
from models import models_expected, model_bkg_flat, model_bkg_exp, model_bkg_refl
from stats import upper_limit

#Genero il toy
mass, cat, label = generate_toy(lo = p.SR_MIN, hi = p.SR_MAX, seed=p.SEED, POI=p.POI, include_signal=True, reflection=True, plot=False, verbose=False)

'''
Binning of the observed datas and expected datas according to three models of background: fixed, exp, exp+refl
'''

obs, bins, bkg_exp_flat, sig_exp, _ = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="flat", PLOT=False
)


_, _, bkg_exp_exp, _, _ = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="exp", PLOT=False
)


_, _, _, _, bkg_exp_ref = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="refl", PLOT=False
)

#creo modelli per singola categoria
ws_cat_flat = []
ws_cat_exp = []
ws_cat_ref = []

for c in range(p.N_CATEGORIES):
	#flat
	ws_cat_f = model_bkg_flat(obs, bkg_exp_flat, sig_exp, cats=[c])
	#exp
	ws_cat_e = model_bkg_exp(obs, bkg_exp_exp, sig_exp, cats=[c])
	#ref
	ws_cat_r = model_bkg_refl(obs, bkg_exp_exp, sig_exp, bkg_exp_ref, cats=[c])
	
	ws_cat_flat.append(ws_cat_f)
	ws_cat_exp.append(ws_cat_e)
	ws_cat_ref.append(ws_cat_r)

#modello con tutte le categorie
ws_tot_flat = model_bkg_flat(obs, bkg_exp_flat, sig_exp, cats=None)
ws_tot_exp  = model_bkg_exp(obs, bkg_exp_exp, sig_exp, cats=None)
ws_tot_ref = model_bkg_refl(obs, bkg_exp_exp, sig_exp, bkg_exp_ref, cats=None)

#calcolo upper limit

##### FOR CATEGORY #####

#I use only grid for the moment
upper_limit_obs_flat = []
upper_limit_obs_exp  = []
upper_limit_obs_refl = []

upper_limit_exp_flat = []
upper_limit_exp_exp  = []
upper_limit_exp_refl = []

#flat
print("UPPER LIMIT FOR SINGLE CATEGORY FOR FLAT BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	obs_flat, exp_flat, _, _ = upper_limit(ws_cat_flat[c])
	upper_limit_obs_flat.append(obs_flat)
	upper_limit_exp_flat.append(exp_flat[2])
		
#exp
print("UPPER LIMIT FOR SINGLE CATEGORY FOR EXP BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	obs_exp, exp_exp, _, _ = upper_limit(ws_cat_exp[c])
	upper_limit_obs_exp.append(obs_exp)
	upper_limit_exp_exp.append(exp_exp[2])

#exp + ref
print("UPPER LIMIT FOR SINGLE CATEGORY FOR EXP+REF BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	obs_refl, exp_refl, _, _ = upper_limit(ws_cat_ref[c])
	upper_limit_obs_refl.append(obs_refl)
	upper_limit_exp_refl.append(exp_refl[2])

print("\n" + "="*40)
print(" RIEPILOGO UPPER LIMIT PER CATEGORIA")
print("="*40)
for c in range(p.N_CATEGORIES):
    print(f"Categoria {c+1}:")
    print(f"  [Flat]    Obs: {upper_limit_obs_flat[c]} | Exp: {upper_limit_exp_flat[c]}")
    print(f"  [Exp]     Obs: {upper_limit_obs_exp[c]} | Exp: {upper_limit_exp_exp[c]}")
    print(f"  [Exp+Ref] Obs: {upper_limit_obs_refl[c]} | Exp: {upper_limit_exp_refl[c]}")
    print("-" * 40)

#Sort categories by expected events
expected_events = p.BKG_SR
expected_events_flat = [np.sum(bkg_exp_flat[c]) for c in range(p.N_CATEGORIES)]
expected_events_exp = [np.sum(bkg_exp_exp[c]) for c in range(p.N_CATEGORIES)]
expected_events_refl = [np.sum(bkg_exp_ref[c]) for c in range(p.N_CATEGORIES)]

sorted_indices_flat = np.argsort(expected_events_flat)[::-1]
sorted_indices_exp = np.argsort(expected_events_exp)[::-1]
sorted_indices_refl = np.argsort(expected_events_refl)[::-1]
sorted_categories_labels_flat = [f"Cat {i+1}" for i in sorted_indices_flat]
sorted_categories_labels_exp = [f"Cat {i+1}" for i in sorted_indices_exp]
sorted_categories_labels_refl = [f"Cat {i+1}" for i in sorted_indices_refl]

x_numeric = range(1, p.N_CATEGORIES + 1)

upper_limit_obs_flat = [upper_limit_obs_flat[i] for i in sorted_indices_flat]
upper_limit_exp_flat = [upper_limit_exp_flat[i] for i in sorted_indices_flat]
upper_limit_obs_exp  = [upper_limit_obs_exp[i] for i in sorted_indices_exp]
upper_limit_exp_exp  = [upper_limit_exp_exp[i] for i in sorted_indices_exp]
upper_limit_obs_refl = [upper_limit_obs_refl[i] for i in sorted_indices_refl]
upper_limit_exp_refl = [upper_limit_exp_refl[i] for i in sorted_indices_refl]

## PLOT ##

plt.figure(figsize=(14, 5))

# Subplot 1: Flat Bkg
plt.subplot(1, 3, 1)
plt.plot(x_numeric, upper_limit_obs_flat, marker='o', color='blue', label='Observed')
plt.plot(x_numeric, upper_limit_exp_flat, marker='s', linestyle='--', color='navy', label='Expected')
plt.title('Flat Background')
plt.xlabel('Categories (Sorted by Expected Events)')
plt.ylabel('Upper Limit')
plt.xticks(x_numeric, sorted_categories_labels_flat, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Exp Bkg
plt.subplot(1, 3, 2)
plt.plot(x_numeric, upper_limit_obs_exp, marker='o', color='darkorange', label='Observed')
plt.plot(x_numeric, upper_limit_exp_exp, marker='s', linestyle='--', color='firebrick', label='Expected')
plt.title('Exp Background')
plt.xlabel('Categories (Sorted by Expected Events)')
plt.xticks(x_numeric, sorted_categories_labels_exp, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 3: Exp + Ref Bkg
plt.subplot(1, 3, 3)
plt.plot(x_numeric, upper_limit_obs_refl, marker='o', color='forestgreen', label='Observed')
plt.plot(x_numeric, upper_limit_exp_refl, marker='s', linestyle='--', color='purple', label='Expected')
plt.title('Exp + Ref Background')
plt.xlabel('Categories (Sorted by Expected Events)')
plt.xticks(x_numeric, sorted_categories_labels_refl, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

##### FOR TOTAL ######
print("UPPER LIMIT FOR TOT FOR FLAT BKG")
upper_limit(ws_tot_flat, PLOT = True)
print("UPPER LIMIT FOR TOT FOR EXP BKG")
upper_limit(ws_tot_exp, PLOT = True)
print("UPPER LIMIT FOR TOT FOR REF BKG")
upper_limit(ws_tot_ref, PLOT = True)

