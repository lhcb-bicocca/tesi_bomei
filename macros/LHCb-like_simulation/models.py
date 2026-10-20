import numpy as np
import pyhf
from scipy.stats import johnsonsu

import parameters as p
from toy_generator import xs, cdf

# Binning and analytic integrals
def get_exp_binned(
    bins, 
    expected_total, 
    slope, 
    lo = p.MASS_MIN, 
    hi = p.MASS_MAX
):
    if abs(slope) < 1e-10:
        total_width = hi - lo
        return expected_total * np.diff(bins) / total_width
    
    norm_tot = (np.exp(slope * hi) - np.exp(slope * lo)) / slope
    cdf_vals = (np.exp(slope * bins) - np.exp(slope * lo)) / slope
    bin_fractions = np.diff(cdf_vals) / norm_tot
    return expected_total * bin_fractions

def get_johnson_binned(
    bins, 
    year, 
    loc, 
    scale, 
    expected_total, 
    q0_dict, 
    q1_dict, 
    dq0 = 0.0, 
    dq1 = 0.0
):
    q0 = q0_dict[year] + dq0
    q1 = q1_dict[year] + dq1
    d = johnsonsu(q0, q1, loc=loc, scale=scale)
    cdf_lo = d.cdf(p.MASS_MIN)
    cdf_hi = d.cdf(p.MASS_MAX)
    norm_tot = cdf_hi - cdf_lo
    if norm_tot <= 0: 
        norm_tot = 1.0
    
    cdf_bins = d.cdf(bins)
    bin_fractions = np.diff(cdf_bins) / norm_tot
    res = expected_total * bin_fractions
    return np.maximum(res, 1e-300)

def get_signal_binned(
    bins, 
    year, 
    expected_total, 
    dmean = 0.0, 
    dlambda = 0.0, 
    dgamma = 0.0, 
    ddelta = 0.0
):
    gamma = p.SIGNAL_GAMMA[year] + dgamma
    delta = p.SIGNAL_DELTA[year] + ddelta
    loc = p.SIGNAL_MEAN_CALIB[year] + dmean
    scale = p.SIGNAL_LAMBDA_CALIB[year] + dlambda
    
    d = johnsonsu(gamma, delta, loc=loc, scale=scale)
    cdf_lo = d.cdf(p.MASS_MIN)
    cdf_hi = d.cdf(p.MASS_MAX)
    norm_tot = cdf_hi - cdf_lo
    if norm_tot <= 0: 
        norm_tot = 1.0
        
    cdf_bins = d.cdf(bins)
    bin_fractions = np.diff(cdf_bins) / norm_tot
    res = expected_total * bin_fractions
    return np.maximum(res, 1e-300)

def get_dseta_binned(
    bins, 
    expected_total
):
    c_lo = np.interp(p.MASS_MIN, xs, cdf)
    c_hi = np.interp(p.MASS_MAX, xs, cdf)
    norm_tot = c_hi - c_lo
    
    cdf_bins = np.interp(bins, xs, cdf)
    bin_fractions = np.diff(cdf_bins) / norm_tot
    return expected_total * bin_fractions

# Uncertainties of the signal shape parameters
def signal_shape_sigmas(
    year
):
    sigma_mean = np.hypot(
        p.SIGNAL_MEAN_CORR[year] * p.SIGNAL_MEAN_ERRS[year],
        p.SIGNAL_MEAN[year] * p.SIGNAL_MEAN_CORR_ERRS[year]
    )
    sigma_lambda = np.hypot(
        p.SIGNAL_LAMBDA_CORR[year] * p.SIGNAL_LAMBDA_ERRS[year],
        p.SIGNAL_LAMBDA[year] * p.SIGNAL_LAMBDA_CORR_ERRS[year]
    )
    return {
        "mean": sigma_mean, 
        "lambda": sigma_lambda,
        "gamma": p.SIGNAL_GAMMA_ERRS[year], 
        "delta": p.SIGNAL_DELTA_ERRS[year]
    }

# pyhf workspace construction
def build_workspace(
    x, 
    cat, 
    label, 
    model_type = "exp", 
    exclude_cat = [], 
    signal_syst = False,
    misid_syst = False, 
    use_staterror = False,
    slope_prior_width = 0.005, #da sistemare
    constrain_yields = True
):
    bin_width = p.BIN_WIDTH
    n_bins = int(round((p.MASS_MAX - p.MASS_MIN) / bin_width))
    bins = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
    
    def yield_modifier(name, rel_err):
        # Gaussian constraint (normsys) or free (normfactor)
        if constrain_yields:
            return {"name": name, "type": "normsys", "data": {"lo": 1.0 - rel_err, "hi": 1.0 + rel_err}}
        return {"name": name, "type": "normfactor", "data": None}
        
    channels = []
    observations = []
    
    for iy, y in enumerate(p.YEARS):
        for c in range(p.CATEGORIES):
            ic = iy * p.CATEGORIES + c  
            x_cat = x[cat == ic]
            
            if c in exclude_cat:
                continue
  
            #skip channels with 0 expected background, otherwise crash
            if p.C_BKG_YIELDS[y][c] <= 0:
                if x_cat.size > 0:
                    raise ValueError(f"data in channel year_{y}_cat_{c} with null expected yield")
                continue
            
            cat_name = f"year_{y}_cat_{c}"
            
            
            counts, _ = np.histogram(x_cat, bins=bins)
            observations.append({
                "name": cat_name, 
                "data": counts.tolist()
            })
            
            samples = []
            
            # SIGNAL
            sig_yield = p.BR_REF * p.SIGNAL_EFF[y][c] / p.ALPHA[y]
            sig_data = get_signal_binned(bins, y, sig_yield)
            sig_data = np.maximum(sig_data, 1e-6)

            sig_modifiers = [
                # POI
                {
                    "name": "mu", 
                    "type": "normfactor", 
                    "data": None
                },
                # SINGLE_EVENT_SENSITIVITY
                {
                    "name": f"alpha_unc_{y}", 
                    "type": "normsys", 
                    "data": {
                        "lo": p.ALPHA_LO[y], 
                        "hi": p.ALPHA_HI[y]
                    }
                }
            ]
            
            if signal_syst:
                # EFFICIENCY
                if p.SIGNAL_EFF[y][c] > 0: #I efficiency must be positive
                    rel = p.SIGNAL_EFF_ERRS[y][c] / p.SIGNAL_EFF[y][c]
                else:
                    rel = 0.0
                
                sig_modifiers.append({
                    "name": f"sig_eff_{y}_{c}", 
                    "type": "normsys", 
                    "data": {
                        "lo": 1.0 - rel, 
                        "hi": 1.0 + rel
                    }
                })
                
                # SHAPE_PARAMETERS
                sig = signal_shape_sigmas(y)
                for par in ["mean", "lambda", "gamma", "delta"]:
                    up = get_signal_binned(
                        bins, y, sig_yield, **{f"d{par}": + sig[par]}
                    )
                    down = get_signal_binned(
                        bins, y, sig_yield, **{f"d{par}": - sig[par]}
                    )
                    sig_modifiers.append({
                        "name": f"sig_{par}_{y}", 
                        "type": "histosys", 
                        "data": {
                            "lo_data": down.tolist(), 
                            "hi_data": up.tolist()
                        }
                    })
            
            if use_staterror:
                # POISSON
                sig_modifiers.append({
                    "name": f"stat_signal_{cat_name}",
                    "type": "staterror",
                    "data": np.sqrt(np.maximum(sig_data, 1e-9)).tolist()
                })
            
            samples.append({
                "name": "signal", 
                "data": sig_data.tolist(), 
                "modifiers": sig_modifiers
            })
           
            # COMBINATORIAL BACKGROUND
            comb_yield = p.C_BKG_YIELDS[y][c]
            slope_nom = 0.0 if model_type == "flat" else p.SLOPE_BKG_VALS[y][c]
            comb_data = get_exp_binned(
                bins, 
                comb_yield, 
                slope_nom, 
                p.MASS_MIN, 
                p.MASS_MAX
            )
           
            bkg_modifiers = [
            
                # NORMALIZATION
                {
                    "name": f"bkg_norm_{cat_name}", 
                    "type": "normfactor", 
                    "data": None
                }
            ]
            
           
            if model_type != "flat": 
                comb_up = get_exp_binned(
                    bins, 
                    comb_yield, 
                    slope_nom + slope_prior_width, 
                    p.MASS_MIN, 
                    p.MASS_MAX
                )
                comb_down = get_exp_binned(
                    bins, 
                    comb_yield, 
                    slope_nom - slope_prior_width, 
                    p.MASS_MIN, 
                    p.MASS_MAX
                )
               
                bkg_modifiers.append({
                    # SLOPE UNCERTAINTY
                    "name": f"bkg_slope_{cat_name}",
                    "type": "histosys",
                    "data": {
                        "lo_data": comb_down.tolist(), 
                        "hi_data": comb_up.tolist()
                    }
                })
            
            if use_staterror:
                bkg_modifiers.append({
                    "name": f"stat_comb_{cat_name}",
                    "type": "staterror",
                    "data": np.sqrt(np.maximum(comb_data, 1e-9)).tolist()
                })
           
            samples.append({
                "name": "background_combinatorial",
                "data": comb_data.tolist(),
                "modifiers": bkg_modifiers
            })
            
            # MISID 
            if model_type in ["exp_misid", "exp_misid_dseta"]:
            
                # Dp
                dp_yield = p.DP_YIELDS[y] * p.DP_EFF[y][c]
                dp_err = p.DP_YIELDS_ERRS[y]
                
                dp_data = get_johnson_binned(
                    bins, 
                    y, 
                    p.DP_MEAN_CALIB[y], 
                    p.DP_SIGMA[y], 
                    dp_yield, 
                    p.MISID_TAIL_Q0, 
                    p.MISID_TAIL_Q1
                )
                
                rel_dp = p.DP_YIELDS_ERRS[y] / p.DP_YIELDS[y]
                if rel_dp >= 1:
                    raise ValueError(f"relative uncertainty Ds yield {y} >= 1")
                
                dp_modifiers = [yield_modifier(f"dp_yield_{y}", rel_dp)]
                
                if misid_syst:
                    q0_err = p.MISID_TAIL_Q0_ERRS[y]
                    q1_err = p.MISID_TAIL_Q1_ERRS[y]
                    
                    dp_up_q0 = get_johnson_binned(
                        bins, y, p.DP_MEAN_CALIB[y], p.DP_SIGMA[y], 
                        dp_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq0 = +q0_err
                    )
                    dp_down_q0 = get_johnson_binned(
                        bins, y, p.DP_MEAN_CALIB[y], p.DP_SIGMA[y], 
                        dp_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq0 = -q0_err
                    )
                    dp_modifiers.append({
                        "name": f"misid_q0_{y}",
                        "type": "histosys",
                        "data": {
                            "lo_data": dp_down_q0.tolist(), 
                            "hi_data": dp_up_q0.tolist()
                        }
                    })

                    dp_up_q1 = get_johnson_binned(
                        bins, y, p.DP_MEAN_CALIB[y], p.DP_SIGMA[y], 
                        dp_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq1 = +q1_err
                    )
                    dp_down_q1 = get_johnson_binned(
                        bins, y, p.DP_MEAN_CALIB[y], p.DP_SIGMA[y], 
                        dp_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq1 = -q1_err
                    )
                    dp_modifiers.append({
                        "name": f"misid_q1_{y}",
                        "type": "histosys",
                        "data": {
                            "lo_data": dp_down_q1.tolist(), 
                            "hi_data": dp_up_q1.tolist()
                        }
                    })
                
                if use_staterror:
                    dp_modifiers.append({
                        "name": f"stat_dp_{cat_name}",
                        "type": "staterror",
                        "data": np.sqrt(np.maximum(dp_data, 1e-9)).tolist()
                    })

                samples.append({
                    "name": "bkg_Dp", 
                    "data": dp_data.tolist(), 
                    "modifiers": dp_modifiers
                })
                
                # Ds
                ds_yield = p.DS_YIELDS[y] * p.DS_EFF[y][c]
                ds_err = p.DP_YIELDS_ERRS[y]
                ds_data = get_johnson_binned(
                    bins, 
                    y, 
                    p.DS_MEAN_CALIB[y], 
                    p.DS_SIGMA[y], 
                    ds_yield, 
                    p.MISID_TAIL_Q0, 
                    p.MISID_TAIL_Q1
                )
                
                rel_ds = p.DS_YIELDS_ERRS[y] / p.DS_YIELDS[y]
                if rel_ds >= 1:
                    raise ValueError(f"relative uncertainty on Ds yield {y} >= 1")
                ds_modifiers = [yield_modifier(f"ds_yield_{y}", rel_ds)]
                
                if misid_syst:
                    q0_err = p.MISID_TAIL_Q0_ERRS[y]
                    q1_err = p.MISID_TAIL_Q1_ERRS[y]
                    
                    ds_up_q0 = get_johnson_binned(
                        bins, y, p.DS_MEAN_CALIB[y], p.DS_SIGMA[y], 
                        ds_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq0 = +q0_err
                    )
                    ds_down_q0 = get_johnson_binned(
                        bins, y, p.DS_MEAN_CALIB[y], p.DS_SIGMA[y], 
                        ds_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq0 = -q0_err
                    )
                    ds_modifiers.append({
                        "name": f"misid_q0_{y}",
                        "type": "histosys",
                        "data": {
                            "lo_data": ds_down_q0.tolist(), 
                            "hi_data": ds_up_q0.tolist()
                        }
                    })

                    ds_up_q1 = get_johnson_binned(
                        bins, y, p.DS_MEAN_CALIB[y], p.DS_SIGMA[y], 
                        ds_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq1 = +q1_err
                    )
                    ds_down_q1 = get_johnson_binned(
                        bins, y, p.DS_MEAN_CALIB[y], p.DS_SIGMA[y], 
                        ds_yield, p.MISID_TAIL_Q0, p.MISID_TAIL_Q1, dq1 = -q1_err
                    )
                    ds_modifiers.append({
                        "name": f"misid_q1_{y}",
                        "type": "histosys",
                        "data": {
                            "lo_data": ds_down_q1.tolist(), 
                            "hi_data": ds_up_q1.tolist()
                        }
                    })
                
                if use_staterror:
                    ds_modifiers.append({
                        "name": f"stat_ds_{cat_name}",
                        "type": "staterror",
                        "data": np.sqrt(np.maximum(ds_data, 1e-9)).tolist()
                    })

                samples.append({
                    "name": "bkg_Ds", 
                    "data": ds_data.tolist(), 
                    "modifiers": ds_modifiers
                })

            # DSETA
            if model_type == "exp_misid_dseta":
                dseta_yield = p.DSETA_YIELDS[y] * p.DSETA_EFF[y][c]
                dseta_err = p.DSETA_YIELDS_ERRS[y]
                dseta_data = get_dseta_binned(bins, dseta_yield)
                
                rel_dseta = p.DSETA_YIELDS_ERRS[y] / p.DSETA_YIELDS[y]
                if rel_dseta >= 1:
                    raise ValueError(f"relative unc Dseta yield {y} >= 1")

                dseta_modifiers = [yield_modifier(f"dseta_yield_{y}", rel_dseta)]
                
                if use_staterror:
                    dseta_modifiers.append({
                        "name": f"stat_dseta_{cat_name}",
                        "type": "staterror",
                        "data": np.sqrt(np.maximum(dseta_data, 1e-9)).tolist()
                    })

                samples.append({
                    "name": "bkg_Dseta", 
                    "data": dseta_data.tolist(), 
                    "modifiers": dseta_modifiers
                })

            channels.append({
                "name": cat_name, 
                "samples": samples
            })

    spec = {
        "channels": channels,
        "observations": observations,
        "measurements": [
            {
                "name": "tau3mu_measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {
                            "name": "mu", 
                            "bounds": [[0, 100]]
                        }
                    ]
                }
            }
        ],
        "version": "1.0.0"
    }
    
    return pyhf.Workspace(spec)
