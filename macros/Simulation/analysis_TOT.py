import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.contrib.viz import brazil

import parameters as p
from toy_generator import generate_toy
from models import models_expected, model_bkg_flat, model_bkg_exp, model_bkg_refl
from stats import upper_limit

#Genero il toy
mass, cat, label = generate_toy(lo = p.MASS_MIN, hi = p.MASS_MAX, seed=p.SEED, POI=p.POI, include_signal=True, reflection=True, plot=False, verbose=False)

'''
Binning of the observed datas and expected datas according to three models of background: fixed, exp, exp+refl
'''

obs, bins, bkg_exp_flat, sig_exp, _ = models_expected(
    mass, cat, lo=p.MASS_MIN, hi=p.MASS_MAX, model="flat", PLOT=False
)


_, _, bkg_exp_exp, _, _ = models_expected(
    mass, cat, lo=p.MASS_MIN, hi=p.MASS_MAX, model="exp", PLOT=False
)


_, _, _, _, bkg_exp_ref = models_expected(
    mass, cat, lo=p.MASS_MIN, hi=p.MASS_MAX, model="refl", PLOT=False
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

#flat
print("UPPER LIMIT FOR SINGLE CATEGORY FOR FLAT BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	upper_limit(ws_cat_flat[c])
	
#exp
print("UPPER LIMIT FOR SINGLE CATEGORY FOR EXP BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	upper_limit(ws_cat_exp[c])

#exp + ref
print("UPPER LIMIT FOR SINGLE CATEGORY FOR EXP+REF BKG")
for c in range(p.N_CATEGORIES):
	print(f"--- Category {c+1} ---")
	upper_limit(ws_cat_ref[c])

##### FOR TOTAL ######
print("UPPER LIMIT FOR TOT FOR FLAT BKG")
upper_limit(ws_tot_flat)
print("UPPER LIMIT FOR TOT FOR EXP BKG")
upper_limit(ws_tot_exp)
print("UPPER LIMIT FOR TOT FOR REF BKG")
upper_limit(ws_tot_ref)











	


