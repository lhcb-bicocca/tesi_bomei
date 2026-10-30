'''
mc_plots.py

Plots and numbers of the Monte Carlo of monte_carlo.py (observed limit on many toys).

Usage:   python mc_plots.py path_to_csv [tag]

Output in plots/monte_carlo/:
 1. mc_obs_<tag>.pdf      observed limit: distribution, median and +-1/+-2 sigma bands from the quantiles, median of the expected limit
 2. mc_exp_<tag>.pdf      expected limit (one value per toy)
 3. mc_scatter_<tag>.pdf  observed vs expected limit of each toy
 4. mc_pull_<tag>.pdf     pull (mu_free - mu_inj) / sigma, with gaussian fit and N(0, 1)
 5. mc_plots_<tag>.txt    the numbers
'''

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm

import mypkg.parameters as p

path = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(path))[0]
MU_INJ = float(os.environ.get("MU_INJ", 0.0))
QUANTILES = [2.275, 15.865, 50.0, 84.135, 97.725] # -2s, -1s, median, +1s, +2s
BAND_COLORS = ["#ffff00", "#00aa00"] # 2 sigma yellow, 1 sigma green
UNIT = 1e-8 # BR in units of 1e-8
PLOTS_DIR = os.path.join("plots", "monte_carlo")
DATA_DIR = "data"
os.makedirs(PLOTS_DIR, exist_ok = True)
os.makedirs(DATA_DIR, exist_ok = True)

with open(path) as f:
    all_rows = list(csv.DictReader(f))
# Check for required columns
if not all_rows or not {"exp_br"} <= set(all_rows[0]):
    sys.exit(f"'{path}' is empty or has an old format (no exp_br / pull columns): rerun monte_carlo.py on a new file")
    
success = {int(r["seed"]): r for r in all_rows if r["status"] == "ok"}             
n_fail = len({int(r["seed"]) for r in all_rows}) - len(success)
col = lambda name: np.array([float(r[name]) for r in success.values()])
ul_u, ul_br, exp_br, pull = col("ul"), col("ul_br") / UNIT, col("exp_br") / UNIT, col("pull")    # ul_u: in units of BR_REF
n = len(ul_br)

out = open(os.path.join(PLOTS_DIR, f"mc_plots_{tag}.txt"), "w", encoding = "utf-8")
def log_print(text = ""):
    print(text)
    out.write(text + "\n")

log_print(f"{path}: {n} toys ok, {n_fail} failed, injected mu = {MU_INJ:g}, limits at {(1 - p.CL) * 100:.0f}% CL")
log_print(f"  seconds per toy: median = {np.median(col('seconds')):.0f}")

has_exp = np.isfinite(exp_br).all()
q_obs = np.percentile(ul_br, QUANTILES)
q_exp = np.percentile(exp_br, QUANTILES) if has_exp else None
for name, v in [("BR_obs", q_obs), ("BR_exp", q_exp)]:
    if v is not None:
        log_print(f"  {name} [1e-8]: median = {v[2]:.3f}   -2s = {v[0]:.3f}  -1s = {v[1]:.3f}  +1s = {v[3]:.3f}  +2s = {v[4]:.3f}")
if not has_exp:
    log_print("  BR_exp: not available (monte_carlo.py run with WITH_EXP=0)")

def bands(ax, q):
    ax.axvspan(q[0], q[4], color = BAND_COLORS[0], alpha = 0.5, lw = 0)
    ax.axvspan(q[1], q[3], color = BAND_COLORS[1], alpha = 0.5, lw = 0)
    ax.axvline(q[2], color = "k", lw = 2, zorder = 4)

def distribution(values, q, xlabel, filename, extra = None):
    fig, ax = plt.subplots(figsize = (9, 6))
    ax.hist(values, bins = max(10, int(np.sqrt(len(values)) * 1.5)), color = "#4f81bd", edgecolor = "black", linewidth = 0.6, zorder = 3)
    bands(ax, q)
    if extra is not None:
        ax.axvline(extra[0], color = "#c0504d", lw = 2, ls = "--", label = extra[1], zorder = 4)
    ax.plot([], [], color = "k", lw = 2, label = f"median = {q[2]:.2f}")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color = BAND_COLORS[1], alpha = 0.5, label = r"$\pm 1\sigma$ (15.9-84.1%)"))
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color = BAND_COLORS[0], alpha = 0.5, label = r"$\pm 2\sigma$ (2.3-97.7%)"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Toys (N = {len(values)})")
    ax.legend(frameon = False, loc = "lower center", bbox_to_anchor = (0.5, 1.0), ncol = 2, fontsize = 13)
    fig.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches = "tight")
    plt.close(fig)

with plt.style.context(hep.style.LHCb2):
    cl = f"{(1 - p.CL) * 100:.0f}% CL"

    # 1. BR_obs (with the median of BR_exp)
    extra = (q_exp[2], "median of BR_exp") if has_exp else None
    distribution(ul_br, q_obs, rf"$BR_{{obs}}$ [$10^{{-8}}$] ({cl})", f"mc_obs_{tag}.pdf", extra = extra)

    if has_exp:
        # 2. BR_exp
        distribution(exp_br, q_exp, rf"$BR_{{exp}}$ [$10^{{-8}}$] ({cl})", f"mc_exp_{tag}.pdf")

        # 3. scatter (independent axes: BR_exp varies by a few %, BR_obs by tens of %)
        r = np.corrcoef(ul_br, exp_br)[0, 1]
        spread_exp = (q_exp[3] - q_exp[1]) / 2 / q_exp[2] * 100
        spread_obs = (q_obs[3] - q_obs[1]) / 2 / q_obs[2] * 100
        fig, ax = plt.subplots(figsize = (8, 7))
        ax.scatter(exp_br, ul_br, s = 18, color = "#4f81bd", edgecolor = "black", linewidth = 0.4, zorder = 3)
        margin = 0.15 * max(exp_br.max() - exp_br.min(), 1e-9)
        ax.set_xlim(exp_br.min() - margin, exp_br.max() + margin)
        ax.axhline(q_obs[2], color = "gray", lw = 0.8, ls = "--", label = "median of BR_obs")
        ax.axvline(q_exp[2], color = "gray", lw = 0.8, label = "median of BR_exp")
        ax.set_xlabel(rf"$BR_{{exp}}$ [$10^{{-8}}$] ({cl}, per toy)")
        ax.set_ylabel(rf"$BR_{{obs}}$ [$10^{{-8}}$] ({cl})")
        ax.text(0.04, 0.95, f"r = {r:+.2f}  (N = {n})\nspread BR_exp: $\\pm${spread_exp:.1f}%\nspread BR_obs: $\\pm${spread_obs:.0f}%",
                transform = ax.transAxes, fontsize = 13, va = "top")
        ax.legend(frameon = False, loc = "lower right", fontsize = 12)
        fig.savefig(os.path.join(PLOTS_DIR, f"mc_scatter_{tag}.pdf"), bbox_inches = "tight")
        plt.close(fig)
        log_print(f"  correlation BR_obs - BR_exp: r = {r:+.3f}; spread (half of the +-1s band) BR_exp = {spread_exp:.1f}%, BR_obs = {spread_obs:.0f}%")

    # 4. pull
    ok = np.isfinite(pull)
    if ok.sum() > 1:
        pl = pull[ok]
        mean, std = pl.mean(), pl.std(ddof = 1)
        fig, ax = plt.subplots(figsize = (9, 6))
        edges = np.linspace(-4, 4, 21)
        ax.hist(np.clip(pl, -3.99, 3.99), bins = edges, density = True, color = "#4f81bd", edgecolor = "black", linewidth = 0.6,
                label = f"pull (N = {len(pl)})", zorder = 3)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, norm.pdf(xs, mean, std), "r-", lw = 2, label = rf"Gaussian: mean = {mean:+.2f}, $\sigma$ = {std:.2f}")
        ax.plot(xs, norm.pdf(xs), "k--", lw = 1.5, label = r"$\mathcal{N}(0,1)$")
        ax.set_xlabel(r"$(\hat{\mu} - \mu_{inj}) / \sigma_{\hat{\mu}}$")
        ax.set_ylabel("Density")
        ax.legend(frameon = False, loc = "lower center", bbox_to_anchor = (0.5, 1.0), ncol = 1, fontsize = 13)
        fig.savefig(os.path.join(PLOTS_DIR, f"mc_pull_{tag}.pdf"), bbox_inches = "tight")
        plt.close(fig)
        log_print(f"  pull: mean = {mean:+.3f} +/- {std / np.sqrt(len(pl)):.3f}, std = {std:.3f} +/- {std / np.sqrt(2 * (len(pl) - 1)):.3f}  (expected 0 and 1; {(~ok).sum()} toys without pull)")
    else:
        log_print("  pull: not available (monte_carlo.py run with WITH_PULL=0, or fits failed)")

if MU_INJ > 0:
    log_print(f"  coverage with injected mu = {MU_INJ:g}: fraction of limits > mu_inj = {(ul_u > MU_INJ).mean():.3f} (expected >= {1 - p.CL:.2f})")
log_print(f"\nPlots saved in: {PLOTS_DIR}/ (files *_{tag}.pdf); numbers in mc_plots_{tag}.txt")
out.close()
