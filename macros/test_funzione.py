from mypkg.CLs_library import compute_cls
import pyhf 

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
                        {"name": "mu", "type": "normfactor", "data": None}
                    ]
                }
            ]
        }
    ]
}

model = pyhf.Model(spec, poi_name = "mu")	

compute_cls(
    calculator="toys",
    n_obs=n_obs,
    b=b,
    s=s,
    mu=mu_test,
    PRINT=True
)

compute_cls(
    calculator="poisson",
    n_obs=n_obs,
    b=b,
    s=s,
    mu=mu_test,
    PRINT=True
)

compute_cls(
    calculator="asymptotics",
    n_obs=n_obs,
    b=b,
    s=s,
    mu=mu_test,
    PRINT=True
)

compute_cls(
    calculator="pyhf",
    model = model,
    n_obs=n_obs,
    mu=mu_test,
    calctype = "toys",
    PRINT=True
)

compute_cls(
    calculator="pyhf",
    model = model,
    n_obs=n_obs,
    mu=mu_test,
    calctype = "asymptotics",
    PRINT=True
)


