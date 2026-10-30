'''
models.py

This function build the whole JSON.
The idea is using the parameters from the fit of LHCb, while for the signal the single sensitivity is multiplied.
It computes systematics (mainly normsys, histosys and normfactor) for all the 16 x 3 category bins. 

Something about the parameters in the function (each are chosen in parameters.py):
    - model_type: choose which model to use (base it's exp + MisId + DsEta),
      but bkg can be chosen as flat or exp without MisId and DsEta 
    - exclude_cat = exclude some categories (default: None)
    - signal_sys and misid_sys: systematics on the shape parameters for signal
      and Misid; default is True
    - use_staterror: introduce MC errors; default is False (number of   
      parameters explode) 
    - slope_prior_width: multiplicative factor for slope
    - constrain_yield: normalization for misid and dseta constrained (normsys) 
      or free (normfactor); default is True
    - region = region of interest; Signal, TOT or Sidebands
    - eps_{} = multiplicative factor for the efficiencies
'''

import numpy as np
import pyhf
import mypkg.parameters as p
from mypkg.utils import get_exp_binned, get_johnson_binned, get_signal_binned, get_dseta_binned, signal_shape_sigmas

# pyhf workspace construction
def build_workspace(
    x, 
    cat, 
    label, 
    model_type = p.MODEL, 
    exclude_cat = p.EXCLUDE_CAT, 
    signal_syst = p.SIGNAL_SYST,
    misid_syst = p.MISID_SYST, 
    use_staterror = p.STATERROR,
    slope_prior_width = 1, #da sistemare
    constrain_yields = p.CONSTRAIN_YIELD,
    region = p.REGION,
    sr_constraints = None,
    eps_sig = p.EPS_SIG,       
    eps_comb = p.EPS_COMB,      
    eps_dp = p.EPS_DP,
    eps_ds = p.EPS_DS,
    eps_dseta = p.EPS_DSETA
):
    if region not in ("TOT", "SR", "SB"):
        raise ValueError("region must be 'TOT', 'SR' or 'SB'")
    bin_width = p.BIN_WIDTH
    n_bins = int(round((p.MASS_MAX - p.MASS_MIN) / bin_width))
    bins = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
    
    centers = 0.5 * (bins[1:] + bins[:-1])
    in_sr = (centers >= p.SR_MIN) & (centers <= p.SR_MAX)
    sel = {"TOT": np.ones(n_bins, dtype = bool), "SR": in_sr, "SB": ~in_sr}[region]
    
    def restrict(sample):
        # keep only the bins in 'sel' 
        pick = lambda values: [v for v, k in zip(values, sel) if k]
        modifiers = []
        for m in sample["modifiers"]:
            m = dict(m)
            if m["type"] == "histosys":
                m["data"] = {"lo_data": pick(m["data"]["lo_data"]), "hi_data": pick(m["data"]["hi_data"])}
            elif m["type"] == "staterror":
                m["data"] = pick(m["data"])
            modifiers.append(m)
        return {"name": sample["name"], "data": pick(sample["data"]), "modifiers": modifiers}
    
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
                "data": counts[sel].tolist()
            })
            
            samples = []
            
            # SIGNAL
            sig_yield = (p.BR_REF * p.SIGNAL_EFF[y][c] / p.ALPHA[y]) * eps_sig
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
            comb_yield = p.C_BKG_YIELDS[y][c] * eps_comb
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
                slope_width = slope_prior_width * p.SLOPE_BKG_ERRS[y][c]
                comb_up = get_exp_binned(
                    bins, 
                    comb_yield, 
                    slope_nom + slope_width, 
                    p.MASS_MIN, 
                    p.MASS_MAX
                )
                comb_down = get_exp_binned(
                    bins, 
                    comb_yield, 
                    slope_nom - slope_width, 
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
                dp_yield = (p.DP_YIELDS[y] * p.DP_EFF[y][c]) * eps_dp
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
                ds_yield = (p.DS_YIELDS[y] * p.DS_EFF[y][c]) * eps_ds
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
                dseta_yield = (p.DSETA_YIELDS[y] * p.DSETA_EFF[y][c]) * eps_dseta
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
                
            if region != "TOT":
                samples = [restrict(s) for s in samples]
            
            if region == "SR" and sr_constraints is not None:
                # combinatorial background of the SR from the sideband fit: extrapolated yield and its uncertainty
                con = sr_constraints[cat_name]
                bins_sr = bins[np.where(sel)[0][0]: np.where(sel)[0][-1] + 2]
                comb_sr = get_exp_binned(bins_sr, con["n_sr"], con["slope"], bins_sr[0], bins_sr[-1])
                rel = con["sigma"] / con["n_sr"]
                samples = [s for s in samples if s["name"] != "background_combinatorial"]
                samples.append({
                    "name": "background_combinatorial",
                    "data": comb_sr.tolist(),
                    "modifiers": [{"name": f"bkg_norm_{cat_name}", "type": "normsys", "data": {"lo": 1.0 - rel, "hi": 1.0 + rel}}]
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
    
