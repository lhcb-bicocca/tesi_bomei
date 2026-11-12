'''
fast_limit.py

Compute minimization and CLs with a way faster method.
Briefly, it uses JAX to find gradient and hessian fast (autodiff), than I use scipy minimize with L-BFGS-B.
Lastly, it computes asymptotics UL using Cowan & al approximation

'''

import numpy as np
import jax
import jax.numpy as jnp
import pyhf
from scipy.optimize import minimize, toms748
from scipy.stats import norm

import mypkg.parameters as p

pyhf.set_backend("jax", "scipy")

OPTIONS = {"maxiter": 5000, "maxfun": 20000, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 50}

def setup(model, mu_bounds = (0.0, 100.0)):
    #I define init ad boundaries for mu values
    poi = model.config.poi_index
    bounds = np.array(model.config.suggested_bounds(), dtype = float)
    bounds[poi] = mu_bounds
    init = np.array(model.config.suggested_init(), dtype = float)
    init[poi] = max(init[poi], mu_bounds[0])
    
    def nll(pars, data):
        #-2 ln L
        logL = model.logpdf(pars, data)[0]
        return -2.0 * logL
    
    #value and gradiend wrt pars    
    vg = jax.jit(jax.value_and_grad(nll))
    #hessian
    grad = jax.grad(nll)
    hvp = jax.jit(lambda pars, data, v: jax.jvp(lambda q: grad(q, data), (pars,), (v,))[1])

    return {
    	"model": model, 
    	"vg": vg, 
    	"hvp": hvp, 
    	"poi": poi, 
    	"bounds": bounds, 
    	"init": init
    	}

def counts(model, x, cat):
    #Data  of a toy, without rebuilding the workspace.'''
    obs = []
    for name in model.config.channels:
        _, y, _, c = name.split("_")
        ic = p.YEARS.index(y) * p.CATEGORIES + int(c)
        n_bins = model.config.channel_nbins[name]
        bins = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
        obs.append(np.histogram(x[cat == ic], bins = bins)[0])
    return jnp.asarray(np.concatenate(obs + [np.array(model.config.auxdata)]), dtype = float)

def fit(ctx, data, mu = None, init = None):
    #Minimizes -2 ln L.
    b = ctx["bounds"].copy()
    x0 = (ctx["init"] if init is None else init).copy()
    if mu is not None:
        b[ctx["poi"]] = (mu, mu)
        x0[ctx["poi"]] = mu
    x0 = np.clip(x0, b[:, 0], b[:, 1]) #I need clip in case of differents init

    def f(v):
        #Convert jax to numpy, otherwise scipy doesn't work
        val, g = ctx["vg"](v, data)
        val, g = float(val), np.asarray(g)
        #If non finite values, go back
        if not (np.isfinite(val) and np.all(np.isfinite(g))):
            return 1e30, np.zeros_like(x0)
        return val, g
    
    #I need this to scale the norm of the step of the gradient, otherwise I risk to go out of boundaries
    scale = 1.0 / max(np.linalg.norm(f(x0)[1]), 1.0)

    def f_scaled(v):
        val, g = f(v)
        return (val, g) if val >= 1e29 else (scale * val, scale * g)

    options = dict(OPTIONS, gtol = OPTIONS["gtol"] * scale)

    result = minimize(f_scaled, x0, jac = True, method = "L-BFGS-B", bounds = b, options = options)
    for _ in range(3):
        if result.success:
            break
        result2 = minimize(f_scaled, result.x, jac = True, method = "L-BFGS-B", bounds = b, options = options)
        done = result2.success or result2.fun > result.fun - 1e-9 * scale
        result = result2
        if done:
            break
    if not result.success:
        result = minimize(f, result.x if result.fun < 1e29 * scale else x0, jac = True, method = "SLSQP", bounds = b, options = {"maxiter": 1000, "ftol": 1e-10})
        scale = 1.0
    if not result.success:
        print(f"WARNING: fit did not converge ({result.message})")
    return float(result.fun) / scale, result.x

def mu_free(ctx, data, init = None, mu_bounds = (-10.0, 100.0)):
    #mu_hat WITHOUT the mu >= 0 limit and its uncertainty (for the pulls)
    ctx_free = dict(ctx)  
    ctx_free["bounds"] = ctx["bounds"].copy()
    ctx_free["bounds"][ctx["poi"]] = mu_bounds
    nll, best = fit(ctx_free, data, init = init)
    b = jnp.asarray(best)
    H = np.column_stack([np.asarray(ctx["hvp"](b, data, e)) for e in jnp.eye(len(best))])    
    H = 0.5 * (H + H.T)
    try:
        var = 2.0 * np.linalg.inv(H)[ctx["poi"], ctx["poi"]]
    except np.linalg.LinAlgError:
        var = np.nan
    return best[ctx["poi"]], (np.sqrt(var) if var > 0 else np.nan)

def prepare(ctx, data):
    #Quantities independent of mu
    nll_free, best = fit(ctx, data)
    nll_0, pars0 = fit(ctx, data, mu = 0.0, init = best)
    asimov = ctx["model"].expected_data(pars0)
    nll_A_free = float(ctx["vg"](jnp.asarray(pars0), asimov)[0])
    return {
        "data": data, "mu_hat": best[ctx["poi"]], "nll_free": nll_free, "best": best,
        "pars0": pars0, "asimov": asimov, "nll_A_free": nll_A_free,
        "data_fits": {}, "asimov_fits": {}, "n_fits": 2
    }

def conditional(ctx, ds, kind, mu):
    #Fit at fixed mu on data or Asimov dataset, with warm start from the nearest already calculated mu
    cache = ds[kind + "_fits"]
    #I compute only if fit isn't already in cache
    if mu not in cache:
        if cache:
            mu_near = min(cache, key = lambda m: abs(m - mu))
            init = cache[mu_near][1]
        else:
            init = ds["best"] if kind == "data" else ds["pars0"]
        data = ds["data"] if kind == "data" else ds["asimov"]
        cache[mu] = fit(ctx, data, mu = mu, init = init)
        ds["n_fits"] += 1
    return cache[mu][0]

def cls_obs(ctx, ds, mu):
    #Observed CLs (asymptotic, q-tilde) at value mu (in units of BR_REF)
    nll_mu = conditional(ctx, ds, "data", mu)
    nll_mu_A = conditional(ctx, ds, "asimov", mu)
    q = 0.0 if ds["mu_hat"] > mu else max(nll_mu - ds["nll_free"], 0.0)
    qA = max(nll_mu_A - ds["nll_A_free"], 0.0)
    if qA <= 0.0:
        return 1.0
    sq, sqA = np.sqrt(q), np.sqrt(qA)
    t = sq - sqA if sq <= sqA else (q - qA) / (2.0 * sqA)
    CLsb = norm.cdf(-(t + sqA))
    CLb = norm.cdf(-t)
    return float(CLsb / CLb)

def find_limit(f, mu_min, mu_max, rtol):
    #Zero of f(mu) (positive for small mu, decreasing): widen the upper end until the sign changes, then toms748.
    if f(mu_min) <= 0:
        raise ValueError("already excluded at mu_min: lower mu_min")
    hi = 4.0
    while f(hi) > 0:
        if hi >= mu_max:
            raise ValueError("limit is beyond mu_max: raise mu_max")
        hi = min(2 * hi, mu_max)
    return toms748(f, mu_min, hi, rtol = rtol)

def expected_limit_ds(ctx, ds, level = 0.10, rtol = 1e-3, mu_min = 0.05, mu_max = 1000.0):
    #Median expected limit of ONE toy: Asimov dataset with nuisance at their best fit to the data (mu = 0), as pyhf does.
    #It reuses the Asimov fits already in the cache of the observed limit.
    def f(mu):
        qA = max(conditional(ctx, ds, "asimov", float(mu)) - ds["nll_A_free"], 0.0)
        return 2.0 * norm.cdf(-np.sqrt(qA)) - level
    return find_limit(f, mu_min, mu_max, rtol)

def upper_limit_obs(ctx, data, level = 0.10, rtol = 1e-3, mu_min = 0.1, mu_max = 100.0, with_expected = False):
    #Observed upper limit (CLs = level) with toms748, in units of BR_REF.
    #with_expected = True: also the expected limit of this toy (info["expected"], otherwise nan).
    ds = prepare(ctx, data)
    n_eval = [0]

    def f(mu):
        n_eval[0] += 1
        return cls_obs(ctx, ds, float(mu)) - level

    limit = find_limit(f, mu_min, mu_max, rtol)
    expected = expected_limit_ds(ctx, ds, level, rtol) if with_expected else np.nan
    return limit, {"n_fits": ds["n_fits"], "mu_hat": ds["mu_hat"], "n_eval": n_eval[0], "expected": expected, "best": ds["best"]}

def expected_limit(ctx, level = 0.10, rtol = 1e-3, mu_min = 0.05, mu_max = 1000.0):
    #Median expected limit (asymptotic) 
    model, poi = ctx["model"], ctx["poi"]
    pars0 = ctx["init"].copy()
    pars0[poi] = 0.0
    asimov = model.expected_data(pars0)
    nll_free = float(ctx["vg"](jnp.asarray(pars0), asimov)[0])
    fits = {}

    def f(mu):
        mu = float(mu)
        if mu not in fits:
            init = fits[min(fits, key = lambda m: abs(m - mu))][1] if fits else pars0
            fits[mu] = fit(ctx, asimov, mu = mu, init = init)
        qA = max(fits[mu][0] - nll_free, 0.0)
        return 2.0 * norm.cdf(-np.sqrt(qA)) - level

    return find_limit(f, mu_min, mu_max, rtol)
    
