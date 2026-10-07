import numpy as np
import matplotlib.pyplot as plt
import pyhf
from pyhf.contrib.viz import brazil

import parameters as p
from toy_generator import generate_toy
from models import models_expected, model_bkg_flat, model_bkg_exp, model_bkg_refl

'''SR'''



#Genero il toy
mass, cat, label = generate_toy(seed=p.SEED, POI=p.POI, include_signal=True, reflection=True, plot=False)

#Modello fondo piatto
obs_SR, bins_SR, bkg_exp_flat_SR, sig_exp_SR, _ = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="flat", PLOT=False
)

#Modello fondo esponenziale
_, _, bkg_exp_exp, _, _ = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="exp", PLOT=False
)

#Modello fondo exp + reflections
_, _, bkg_exp_ref, _, bkg_exp_ref = models_expected(
    mass, cat, lo=p.SR_MIN, hi=p.SR_MAX, model="refl", PLOT=False
)




ws_cat1 = model_bkg_refl(obs, bkg_exp_exp, sig_exp, bkg_exp_ref, cats=[0])
ws_combined = model_bkg_refl(obs, bkg_exp_exp, sig_exp, bkg_exp_ref, cats=None)










