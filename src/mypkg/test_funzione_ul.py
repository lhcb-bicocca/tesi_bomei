from mypkg.UL_library import upper_limit
import pyhf
import numpy as np

s = 1 	   #segnale atteso
b = 10 	   #background atteso
n_obs = 8    #eventi osservati
mu_test = 1    #Supponiamo che segnale esista

mu_up = upper_limit(n_obs, b, s, calculator = "asymptotics")

print(f"95% CL upper limit: mu < {mu_up:.4f}")

spec = {
    "channels": [
        {
            "name": "single_channel",
            "samples": [
                {
                    "name": "background",
                    "data": [b],
                    "modifiers": []
                },
                {
                    "name": "signal",
                    "data": [s],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None}
                    ]
                }
            ]
        }
    ]
}

model = pyhf.Model(
    spec,
    poi_name="mu",
)

#Calcolo upper limit
scan = np.linspace(0, 10, 100)
obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
    [n_obs], model, scan = scan, level = 0.05, return_results=True
)

print(obs_limit)
print(exp_limits)

