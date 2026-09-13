from mypkg.UL_library import upper_limit
import pyhf
import numpy as np

s = 1 	   #segnale atteso
b = 10 	   #background atteso
n_obs = 8    #eventi osservati
mu_test = 1    #Supponiamo che segnale esista

mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, calculator = "asymptotics")

print(mu_up_obs)
print(mu_up_exp)

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
                        {
                            "name": "mu",
                            "type": "normfactor",
                            "data": None
                        }
                    ]
                }
            ]
        }
    ],
    "parameters": [
        {
            "name": "mu",
            "inits": [1.0],
            "bounds": [[0.0, 30.0]]
        }
    ]
}

model = pyhf.Model(
    spec,
    poi_name="mu"
)

#Calcolo upper limit
scan = np.linspace(0, 20, 201)
obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
    [n_obs], model, scan = scan, level = 0.05, return_results=True
)

print(obs_limit)
print(exp_limits)

