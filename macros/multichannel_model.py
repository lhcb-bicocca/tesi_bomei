import numpy as np
import pyhf
from mypkg.CLs_library import compute_cls

signal     = [1.0, 2.0, 3.0]
background = [10.0, 20.0, 30.0]

# incertezze relative per bin (fattori, non efficienze)
# segnale: 10% su tutti i bin
sig_rel_unc = [0.10, 0.10, 0.10]
# fondo: 0.001/0.01=10%, 0.001/0.02=5%, 0.001/0.03=3.33%
bkg_rel_unc = [0.10, 0.05, 0.0333]

# template variati ±1σ per histosys
signal_hi = [s * (1 + u) for s, u in zip(signal, sig_rel_unc)]
signal_lo = [s * (1 - u) for s, u in zip(signal, sig_rel_unc)]
bkg_hi    = [b * (1 + u) for b, u in zip(background, bkg_rel_unc)]
bkg_lo    = [b * (1 - u) for b, u in zip(background, bkg_rel_unc)]

model = pyhf.Model(
    {
        "channels": [{
            "name": "single_channel",
            "samples": [
                {
                    "name": "background",
                    "data": background,
                    "modifiers": [
                        {
                            "name": "eff_bkg",
                            "type": "histosys",
                            "data": {"hi_data": bkg_hi, "lo_data": bkg_lo}
                        }
                    ]
                },
                {
                    "name": "signal",
                    "data": signal,
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                        {
                            "name": "eff_sig",
                            "type": "histosys",
                            "data": {"hi_data": signal_hi, "lo_data": signal_lo}
                        }
                    ]
                }
            ]
        }],
        "parameters": [
            {"name": "mu", "inits": [1.0], "bounds": [[0.0, 5]]}
        ]
    },
    poi_name="mu",   # <-- QUESTA RIGA
)

print("Parametri del modello:")
for p in model.config.par_order:
    print(" ", p, "->", model.config.par_slice(p))

# ---- osservazioni (Asimov background-only, per ora) ----
obs_data = background   # mu_true = 0, nessuna fluttuazione
mu_test = 1

init_bkg = list(model.config.suggested_init())
init_bkg[model.config.poi_index] = 0.0        # mu = 0 -> background only
obs_data = model.expected_data(init_bkg)      # shape (5,)

# yield nominale (mu=1, tutti i NPs al valore centrale)
init = model.config.suggested_init()
print("\nYield nominali per bin:", model.expected_data(init))

CLs_observed, CLs_expected = compute_cls(
    calculator="pyhf",
    model = model,
    n_obs=obs_data,
    mu=mu_test,
    calctype = "asymptotics",
    PRINT=True
)

obs_limit, exp_limits = pyhf.infer.intervals.upper_limits.upper_limit(
    data=obs_data,
    model=model,
    ,           # scansione sqrt(mu) -> più efficiente vicino a 0
    level=0.05,            # 95% CL
    return_results=True,   # restituisce anche le curve CLs
)

print(f"\nUpper limit osservato su mu (95% CL): {obs_limit:.3f}")

# exp_limits contiene 5 curve: [-2σ, -1σ, mediana, +1σ, +2σ]
labels = ["-2σ", "-1σ", "mediana", "+1σ", "+2σ"]
for lab, val in zip(labels, exp_limits):
    print(f"  atteso {lab}: {val:.3f}")


