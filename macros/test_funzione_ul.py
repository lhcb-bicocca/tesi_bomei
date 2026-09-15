from mypkg.UL_library import upper_limit
import pyhf
import numpy as np

s = 1 	   #segnale atteso
b = 10 	   #background atteso
n_obs = 8    #eventi osservati
mu_test = 1    #Supponiamo che segnale esista

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

mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, 
		calculator = "asymptotics", PRINT = True)

mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, 
		calculator = "toys", PRINT = True)
		
mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, 
		calculator = "poisson", PRINT = True)

mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, model = model,
		calculator = "pyhf", calctype = "asymptotics",  PRINT = True)

mu_up_obs, mu_up_exp = upper_limit(n_obs, b, s, model = model,
		calculator = "pyhf", calctype = "toybased", PRINT = True)

