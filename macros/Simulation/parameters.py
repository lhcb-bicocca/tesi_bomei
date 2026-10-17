import numpy as np

LEVEL = 0.10

'''
PDG 2024: https://pdg.lbl.gov/2024/listings/rpp2024-list-tau.pdf
'''

TAU_MASS       = 1776.93 #Mev
SIGMA_TAU_MASS = 0.09 #Mev

'''
Datas from LHCb experiment Run 2 paper (https://cds.cern.ch/record/2953558)
'''

N_CATEGORIES = 15 #Number of classifier categories
BIN_WIDTH    = 10 #MeV

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


# --- ANNO 2016 ---
bkg_norm_0_vals = np.array([
    1462.4, 658.13, 404.72, 209.57, 890.13, 332.08, 197.01, 111.48, 
    529.37, 206.13, 149.37, 96.915, 365.25, 169.06, 103.02, 92.641
])
bkg_norm_0_errs = np.array([
    39.846, 27.702, 21.092, 16.333, 30.851, 19.725, 14.928, 12.388, 
    23.342, 14.813, 13.025, 11.825, 19.632, 13.901, 11.073, 11.341
])

# --- ANNO 2017 ---
bkg_norm_1_vals = np.array([
    1404.5, 684.21, 417.97, 214.45, 1033.9, 426.57, 243.33, 118.92, 
    716.97, 310.54, 190.20, 113.34, 435.87, 216.64, 137.70, 79.758
])
bkg_norm_1_errs = np.array([
    38.875, 27.894, 21.331, 16.139, 32.985, 21.831, 16.331, 12.564, 
    27.040, 17.955, 14.494, 12.380, 21.307, 15.485, 12.510, 10.681
])

# --- ANNO 2018 ---
bkg_norm_2_vals = np.array([
    1431.1, 642.83, 444.32, 248.37, 937.16, 402.74, 222.33, 138.43, 
    685.91, 292.47, 187.94, 162.59, 396.62, 224.55, 136.79, 95.513
])
bkg_norm_2_errs = np.array([
    40.867, 29.068, 22.367, 18.049, 32.284, 22.284, 16.030, 13.470, 
    26.494, 17.699, 14.271, 13.539, 20.369, 15.517, 12.366, 10.784
])

# Raggruppamento in tuple per comodità
norm_years_vals = (bkg_norm_0_vals, bkg_norm_1_vals, bkg_norm_2_vals)
norm_years_errs = (bkg_norm_0_errs, bkg_norm_1_errs, bkg_norm_2_errs)

'''      
def weighted_mean_norm(vals_list, errs_list):
    """
    Calcola la media pesata e l'incertezza per i yield di fondo (bkg_norm)
    utilizzando i pesi inversi della varianza (1 / sigma^2).
    """
    vals = np.vstack(vals_list)
    errs = np.vstack(errs_list)
    
    # Pesi statistici basati sull'inverso della varianza
    weights = 1.0 / (errs ** 2)
    
    # Media pesata e somma dei pesi lungo l'asse dei 3 anni
    mean_vals, sum_weights = np.average(vals, axis=0, weights=weights, returned=True)
    
    # Incertezza sulla media pesata: 1 / sqrt(somma dei pesi)
    mean_errs = np.sqrt(1.0 / sum_weights)
    
    return mean_vals, mean_errs

# Esecuzione del calcolo
BKG_NORM, SIGMA_BKG_NORM = weighted_mean_norm(norm_years_vals, norm_years_errs)  

BKG_NORM = BKG_NORM[1:]
SIGMA_BKG_NORM = SIGMA_BKG_NORM[1:]
'''

BKG_NORM       = np.vstack(norm_years_vals).sum(axis=0)[1:]
SIGMA_BKG_NORM = np.sqrt((np.vstack(norm_years_errs)**2).sum(axis=0))[1:]

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

'''
MASS_MIN  = 1606.86 - TAU_MASS #+ MASS_BAND
MASS_MAX  = 1976.86 - TAU_MASS #- MASS_BAND
MASS_BAND = MASS_MAX - MASS_MIN
'''

#Shape values; these are assumed from the figures in the paper

# --- ANNO 0 ---
SLOPE_BKG_2016 = np.array([
    0.00023179, -0.00114439, -0.00281418, -0.00181703, 
    -0.000746462, -0.00239885, -0.00379496, -0.00446993, 
    4.8419e-05, -0.00238551, -0.00245748, -0.00367797, 
    -0.000481953, -0.00153681, -0.00447263, -0.00395008
])
SIGMA_SLOPE_BKG_2016 = np.array([
    0.00025494, 0.00040194, 0.00051527, 0.00075794, 
    0.00032779, 0.00058135, 0.00076725, 0.0011404, 
    0.00041479, 0.00069614, 0.00083258, 0.0010979, 
    0.00050743, 0.00078308, 0.0010402, 0.0011319
])

# --- ANNO 1 ---
SLOPE_BKG_2017 = np.array([
    0.00016461, -0.00118167, -0.0011068, -0.00318198, 
    0.00023224, -0.00246158, -0.00241031, -0.00279681, 
    -0.000836283, -0.00294905, -0.00468683, -0.00347184, 
    -0.001238, -0.00426502, -0.00446182, -0.0053202
])
SIGMA_SLOPE_BKG_2017 = np.array([
    0.0002602, 0.00039118, 0.00048745, 0.00075135, 
    0.00029964, 0.00050769, 0.0006517, 0.0010651, 
    0.0003552, 0.00056098, 0.00076228, 0.0010143, 
    0.00046306, 0.0007223, 0.00090432, 0.0012806
])

# --- ANNO 2 ---
SLOPE_BKG_2018 = np.array([
    0.00020058, -0.00213254, -0.00223051, -0.00203792, 
    -0.000596959, -0.00203751, -0.00306079, -0.00516018, 
    -0.000744323, -0.00229045, -0.00309521, -0.00428167, 
    -0.00176417, -0.00338378, -0.00615948, -0.00324279
])
SIGMA_SLOPE_BKG_2018 = np.array([
    0.00026629, 0.00045001, 0.00049153, 0.0007234, 
    0.00032498, 0.000542, 0.00071129, 0.0010364, 
    0.00036378, 0.00058289, 0.00073529, 0.00082524, 
    0.00049079, 0.00068267, 0.00094672, 0.0010801
])

SLOPE_YEARS = (-SLOPE_BKG_2016, -SLOPE_BKG_2017, -SLOPE_BKG_2018)
SIGMA_SLOPE_YEARS = (SIGMA_SLOPE_BKG_2016, SIGMA_SLOPE_BKG_2017, SIGMA_SLOPE_BKG_2018)

def tau_weighted(vals_list, errs_list):
    """
    Calcola la media pesata e l'incertezza per tau (tau = 1 / lambda)
    usando np.average per il calcolo pesato.
    """
    vals = np.vstack(vals_list)
    errs = np.vstack(errs_list)
    
    weights = 1.0 / (errs ** 2)
    
    lambda_mean, sum_weights = np.average(vals, axis=0, weights=weights, returned=True)
    
    # L'incertezza sulla media pesata di lambda è 1 / sqrt(somma dei pesi)
    lambda_err = np.sqrt(1.0 / sum_weights)
    
    # Conversione in tau e propagazione dell'incertezza
    tau_mean = 1.0 / lambda_mean
    tau_err = (tau_mean ** 2) * lambda_err
    
    return tau_mean, tau_err


TAU_BKG, SIGMA_TAU_BKG = tau_weighted(SLOPE_YEARS, SIGMA_SLOPE_YEARS)
TAU_BKG = TAU_BKG[1:]
SIGMA_TAU_BKG = SIGMA_TAU_BKG[1:]
TAU_BKG = np.ones(N_CATEGORIES) * 130
print(TAU_BKG)

SIGMA_SIG = 10

#SIGMA_SIG_REL_UNC = 0.20

REFLECTIONS = [
    {"name": "Dp_pi3",  "mu": 1869.62 - TAU_MASS,  "sigma": 12.0, "frac": 0.03},
    {"name": "Ds_pi3",  "mu": 1968.49 - TAU_MASS, "sigma": 15.0, "frac": 0.01},
]

'''
#Efficiencies
EFF_BKG = rng.uniform(0.5, 1.0, size=N_CATEGORIES)
EFF_SIG = rng.uniform(0.5, 1.0, size=N_CATEGORIES)

W_BKG = EFF_BKG / EFF_BKG.sum()
W_SIG = EFF_SIG / EFF_SIG.sum()

BKG_CAT = BKG_SR_TOT * W_BKG
SIG_CAT = N_SIG * W_SIG
'''
