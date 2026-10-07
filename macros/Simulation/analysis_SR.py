import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.contrib.viz import brazil

import parameters as p
from toy_generator import generate_toy
from models import models_expected, model_bkg_flat, model_bkg_exp, model_bkg_refl

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

#flat
for c in range(p.N_CATEGORIES):
	model = ws_cat_flat[c].model()
	data  = ws_cat_flat[c].data(model)

	ul_obs_grid, ul_exp_grid = pyhf.infer.intervals.upper_limits.upper_limit(data, model, np.linspace(0,50,100), level=p.LEVEL)

	BR_obs_grid = ul_obs_grid * p.BR_REF
	BR_exp_grid = np.asarray(ul_exp_grid) * p.BR_REF

	print(f"UPPER LIMIT CON ASINTOTICO E GRID for category {c+1}")
	print(f"BR_obs (90% CL) = {BR_obs_grid}")
	print("BR expected (90% CL):")
	print(f"  median         = {BR_exp_grid[2]}")
	print(f"  -2σ            = {BR_exp_grid[0]}")
	print(f"  -1σ            = {BR_exp_grid[1]}")
	print(f"  +1σ            = {BR_exp_grid[3]}")
	print(f"  +2σ            = {BR_exp_grid[4]}")
	print("\n")

	#Asymptotic with rootfinder (toms748)
	ul_obs_root, ul_exp_root = pyhf.infer.intervals.upper_limits.upper_limit(data, model, level=p.LEVEL)

	BR_obs_root = ul_obs_root * p.BR_REF
	BR_exp_root = np.asarray(ul_exp_root) * p.BR_REF

	print(f"UPPER LIMIT CON ASINTOTICO E TOMS748 for category {c+1}")
	print(f"BR_obs (90% CL) = {BR_obs_root}")
	print("BR expected (90% CL):")
	print(f"  median         = {BR_exp_root[2]}")
	print(f"  -2σ            = {BR_exp_root[0]}")
	print(f"  -1σ            = {BR_exp_root[1]}")
	print(f"  +1σ            = {BR_exp_root[3]}")
	print(f"  +2σ            = {BR_exp_root[4]}")
	print("\n")
	
	
	#brazil plot
	test_mus = np.linspace(0, 10, 100)
	results = [pyhf.infer.hypotest(m, data, model, test_stat="qtilde", return_expected_set=True) for m in test_mus]

	fig, ax = plt.subplots()
	brazil.plot_results(test_mus*p.BR_REF, results, test_size = p.LEVEL, ax=ax)

	ax.set_xlabel("Br", fontsize=12)
	ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	ax.set_ylim(0, 1.0)

	plt.show()


