'''
monte_carlo.py

Runs Monte Carlo toy simulations for the LHCb-like experiment, handling also signal injection.
Parallelized fit execution across multiple CPU cores, and result aggregation into a CSV file.
'''

import os
import sys
import csv
import time

# 1 thread per process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

MODEL_TYPE = "exp_misid_dseta"
EXCLUDE_CAT = [int(c) for c in os.environ.get("EXCLUDE_CAT", "").split(",") if c.strip() != ""]
MU_INJ = float(os.environ.get("MU_INJ", 0.0))
BIN_WIDTH = float(os.environ["BIN_WIDTH"]) if os.environ.get("BIN_WIDTH") else None
WITH_EXP = os.environ.get("WITH_EXP", "1") == "1" # expected limit of each toy
WITH_PULL = os.environ.get("WITH_PULL", "1") == "1" # mu_hat without the mu >= 0 limit, its uncertainty and the pull (1 more fit + hessian)
COLUMNS = ["seed", "ul", "ul_br", "exp", "exp_br", "mu_hat", "mu_free", "sigma", "pull", "n_fits", "n_eval", "seconds", "status"]
DATA_DIR = "data"

worker = {}     # per worker: module, model, compiled functions

def init_worker():
    import mypkg.fast_limit as fl
    import mypkg.parameters as p
    from mypkg.toy_generator import generate_toy
    from models import build_workspace
    if BIN_WIDTH is not None:
        p.BIN_WIDTH = BIN_WIDTH                       
    x, cat, label = generate_toy(0, exclude_cat = EXCLUDE_CAT, verbose = False)
    model = build_workspace(x, cat, label, model_type = MODEL_TYPE, exclude_cat = EXCLUDE_CAT).model()
    worker.update(fl = fl, p = p, generate_toy = generate_toy, model = model, ctx = fl.setup(model))

def inject_signal(x, cat, seed, mu_inj):
    # signal: Johnson SU per year (same parameters as models.py template), truncated to window, yield per category
    from scipy.stats import johnsonsu
    p = worker["p"]
    xs, cs = [x], [cat]
    for iy, y in enumerate(p.YEARS):
        signal = johnsonsu(p.SIGNAL_GAMMA[y], p.SIGNAL_DELTA[y], loc = p.SIGNAL_MEAN_CALIB[y], scale = p.SIGNAL_LAMBDA_CALIB[y])
        lo, hi = signal.cdf(p.MASS_MIN), signal.cdf(p.MASS_MAX)
        for c in range(p.CATEGORIES):
            if c in EXCLUDE_CAT:
                continue
            ic = iy * p.CATEGORIES + c
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7919, ic]))     
            n = rng.poisson(mu_inj * p.BR_REF * p.SIGNAL_EFF[y][c] / p.ALPHA[y])
            if n > 0:
                xs.append(signal.ppf(rng.uniform(lo, hi, size = n)))
                cs.append(np.full(n, ic))
    return np.concatenate(xs), np.concatenate(cs)

def run_seed(seed):
    t = time.time()
    row = {"seed": seed, "ul": np.nan, "ul_br": np.nan, "exp": np.nan, "exp_br": np.nan, "mu_hat": np.nan, "mu_free": np.nan, "sigma": np.nan, "pull": np.nan, "n_fits": 0, "n_eval": 0, "seconds": 0.0, "status": "ok"}
    try:
        fl, p = worker["fl"], worker["p"]
        x, cat, label = worker["generate_toy"](seed, exclude_cat = EXCLUDE_CAT, verbose = False)
        if MU_INJ > 0:
            x, cat = inject_signal(x, cat, seed, MU_INJ)
        data = fl.counts(worker["model"], x, cat)
        limit, info = fl.upper_limit_obs(worker["ctx"], data, level = p.CL, with_expected = WITH_EXP)
        row.update(ul = limit, ul_br = limit * p.BR_REF, exp = info["expected"], exp_br = info["expected"] * p.BR_REF,
                   mu_hat = info["mu_hat"], n_fits = info["n_fits"], n_eval = info["n_eval"])
        if WITH_PULL:
            try: # a failure here must not lose the limit: the pull stays nan
                mu_free, sigma = fl.mu_free(worker["ctx"], data, init = info["best"])
                row.update(mu_free = mu_free, sigma = sigma, pull = (mu_free - MU_INJ) / sigma)
            except Exception:
                pass
    except Exception as e:
        row["status"] = ("fail: " + repr(e))[:100].replace(",", ";").replace("\n", " ")
    row["seconds"] = round(time.time() - t, 1)
    return row

def summary(path):
    import mypkg.parameters as p
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    result = {int(r["seed"]): r for r in rows}                       
    ul = np.array([float(r["ul"]) for r in result.values()])
    exp = np.array([float(r["exp"]) for r in result.values()])
    mu_hat = np.array([float(r["mu_hat"]) for r in result.values()])
    pull = np.array([float(r["pull"]) for r in result.values()])
    q = np.percentile(ul, [2.275, 15.865, 50, 84.135, 97.725])
    print(f"\n{len(ul)} seeds ok. Observed limit ({(1 - p.CL) * 100:.0f}% CL), in units of BR_REF = {p.BR_REF:g}:")
    print(f"  median = {q[2]:.3f}   -2s = {q[0]:.3f}  -1s = {q[1]:.3f}  +1s = {q[3]:.3f}  +2s = {q[4]:.3f}   mean = {ul.mean():.3f}")
    if np.isfinite(exp).any():
        qe = np.percentile(exp, [2.275, 15.865, 50, 84.135, 97.725])
        spread = (qe[3] - qe[1]) / 2 / qe[2] * 100
        print(f"  expected (per toy): median = {qe[2]:.3f}   -1s = {qe[1]:.3f}  +1s = {qe[3]:.3f}   spread = {spread:.1f}%   r(obs, exp) = {np.corrcoef(ul, exp)[0, 1]:+.2f}")
    ok = np.isfinite(pull)
    if ok.any():
        print(f"  pull ((mu_free - mu_inj) / sigma): N = {ok.sum()}  mean = {pull[ok].mean():+.3f} +/- {pull[ok].std(ddof = 1) / np.sqrt(ok.sum()):.3f}   std = {pull[ok].std(ddof = 1):.3f} (expected 0 and 1)")
    print(f"  mu_hat: mean = {mu_hat.mean():+.3f}, std = {mu_hat.std(ddof = 1):.3f} (mu_hat >= 0: positive by construction)")
    if MU_INJ > 0:
        print(f"  Coverage with injected mu = {MU_INJ:g}: fraction of limits > mu_inj = {(ul > MU_INJ).mean():.3f} (expected >= {1 - p.CL:.2f})")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 2) // 2)
    
    default_csv = f"ul_seeds_{'bkg' if MU_INJ == 0 else f'inj{MU_INJ:g}'}.csv"
    if len(sys.argv) > 4:
        arg_out = sys.argv[4]
        out = arg_out if os.path.dirname(arg_out) else os.path.join(DATA_DIR, arg_out)
    else:
        out = os.path.join(DATA_DIR, default_csv)

    if os.path.exists(out):
        with open(out) as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            sys.exit(f"'{out}' has a different format (columns: {header}): rename it or use another file name.")

    done = set() #Check if toy already done
    if os.path.exists(out):
        with open(out) as f:
            done = {int(r["seed"]) for r in csv.DictReader(f) if r["status"] == "ok"}
            
    todo = [s for s in range(first, first + n_seeds) if s not in done]
    print(f"Model '{MODEL_TYPE}', excluded categories = {EXCLUDE_CAT}, injected mu = {MU_INJ:g}, BIN_WIDTH = {BIN_WIDTH if BIN_WIDTH else 'default'}")
    print(f"Seeds {first}..{first + n_seeds - 1}: {len(done)} already done in '{out}', {len(todo)} to do, {n_workers} workers")
    
    if not todo:
        summary(out)
        return

    new_file = not os.path.exists(out)
    t0 = time.time()
    with open(out, "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = COLUMNS)
        if new_file:
            writer.writeheader()
        pool = ProcessPoolExecutor(max_workers = n_workers, mp_context = multiprocessing.get_context("spawn"), initializer = init_worker)
        try:
            futures = [pool.submit(run_seed, s) for s in todo]
            n_fail = 0
            for k, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                writer.writerow(row)
                f.flush()
                n_fail += row["status"] != "ok"
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)} seeds, {n_fail} failed, {el / 60:.1f} min elapsed, ~{el / k * (len(todo) - k) / 60:.1f} min remaining", flush = True)
        except KeyboardInterrupt:
            print("Interrupted: completed seeds are in the CSV, relaunch the same command to resume.")
            pool.shutdown(wait = False, cancel_futures = True)
            sys.exit(1)
        pool.shutdown()
    summary(out)

if __name__ == "__main__":
    main()
