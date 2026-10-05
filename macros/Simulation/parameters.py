import numpy as np

'''
PDG 2024: https://pdg.lbl.gov/2024/listings/rpp2024-list-tau.pdf
'''

TAU_MASS       = 1776.93 #Mev
SIGMA_TAU_MASS = 0.09 #Mev

'''
Datas from LHCb experiment Run 2 paper (https://cds.cern.ch/record/2953558)
'''

N_CATEGORIES = 15 #Number of classifier categories

#Expected BR from the paper
BR_REF = 1e-8

#Single event sensitivity
ALPHA       = np.array([1.42e-9, 1.21e-9, 1.05e-9])
SIGMA_ALPHA = np.array([0.16e-9, 0.13e-9, 0.11e-9])

ALPHA_EFF   	= 1 / np.sum(1/ALPHA)
SIGMA_ALPHA_EFF = ALPHA_EFF**2 * np.sqrt(np.sum((SIGMA_ALPHA/ALPHA**2)**2))  
ALPHA_REL  	    = SIGMA_ALPHA_EFF / ALPHA_EFF

ALPHA_HI = 1.0 / (1.0 + ALPHA_REL)
ALPHA_LO = 1.0 / (1.0 - ALPHA_REL)

N_SIG = 1.0 * BR_REF / ALPHA_EFF #~25 for mu = 1
N_SIG_C = N_SIG / N_CATEGORIES #segnale atteso per categoria

#SR region
SR_BAND = 20 #Mev
SR_MAX  = + SR_BAND
SR_MIN  = - SR_BAND

#Background SR (From Table 2)
BKG_SR       = np.array([288.3, 187.3, 111.2, 194.5, 110.3, 79.1, 56.5,
		                 129.9, 63.1, 40.8, 31.3, 64.6, 34.0, 33.3, 20.3])

SIGMA_BKG_SR = np.array([7.3, 5.8, 4.6, 6.4, 4.7, 3.7, 3.3, 4.9, 
			             3.5, 2.9, 2.5, 3.8, 2.9, 2.8, 2.3])
			             
BKG_SR_TOT   = BKG_SR.sum()

OBS_SR       = np.array([284, 174, 110, 203, 121, 67, 41,
                         133, 62, 50, 42, 79, 33, 29, 23])
                         
OBS_SR_TOT   = OBS_SR.sum()

'''
Toy settings
'''

SEED = 42
rng = np.random.default_rng(SEED+1)
POI = 0 #bkg only

#Mass window
MASS_BAND = 150 #Mev
MASS_MAX  = + MASS_BAND
MASS_MIN  = - MASS_BAND

#Shape values; these are assumed from the figures in the paper
TAU_BKG   = np.ones(N_CATEGORIES)* 130 #Mev
SIGMA_SIG = 10  #Mev

TAU_BKG_REL_UNC   = 0.20 #suppongo un 20% 
SIGMA_SIG_REL_UNC = 0.20

REFLECTIONS = [
    {"name": "spurious", "mu": 5.0, "sigma": 8.0, "frac": 0.03},
]

#Efficiencies
EFF_BKG = rng.uniform(0.5, 1.0, size=N_CATEGORIES)
EFF_SIG = rng.uniform(0.5, 1.0, size=N_CATEGORIES)

W_BKG = EFF_BKG / EFF_BKG.sum()
W_SIG = EFF_SIG / EFF_SIG.sum()

BKG_CAT = BKG_SR_TOT * W_BKG
SIG_CAT = N_SIG * W_SIG

