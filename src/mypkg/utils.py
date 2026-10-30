'''
utils.py

I put here all the functions regarding sampling, binning and plotting mainly
'''

import numpy as np
from scipy.stats import johnsonsu
import mypkg.parameters as p
import mplhep as hep
import matplotlib.pyplot as plt
import os

'''TOY SAMPLING FUNCTIONS'''
def sample_combinatorial(rng, n, slope):
    if n == 0:
        return np.array([])
    
    u = rng.uniform(0, 1, size=n)
    # if slope ~ 0, I use flat distribution
    if abs(slope) < 1e-10:
        return rng.uniform(p.MASS_MIN, p.MASS_MAX, size=n)
    
    exp_min = np.exp(slope * p.MASS_MIN)
    exp_max = np.exp(slope * p.MASS_MAX)
    
    x = (1.0 / slope) * np.log(exp_min + u * (exp_max - exp_min))
    return x

def sample_johnson(rng, n, year, loc, scale):
    if n == 0:
        return np.array([])
 
    d = johnsonsu(p.MISID_TAIL_Q0[year], p.MISID_TAIL_Q1[year], loc=loc, scale=scale)
    u = rng.uniform(d.cdf(p.MASS_MIN), d.cdf(p.MASS_MAX), size=n)
    return d.ppf(u)
 
def sample_Ds(rng, n, year):
    return sample_johnson(rng, n, year, p.DS_MEAN_CALIB[year], p.DS_SIGMA[year])
 
def sample_Dp(rng, n, year):
    return sample_johnson(rng, n, year, p.DP_MEAN_CALIB[year], p.DP_SIGMA[year])
    
data = np.load(p.DSETA_TEMPLATE)
xs = data["x"]
ys = data["pdf"]
 
dx = np.diff(xs)
cdf = np.concatenate([[0], np.cumsum(0.5 * (ys[1:] + ys[:-1]) * dx)])
cdf /= cdf[-1]

def sample_Dseta(rng, n):
    if n == 0:
        return np.array([])
 
    c_lo = np.interp(p.MASS_MIN, xs, cdf)
    c_hi = np.interp(p.MASS_MAX, xs, cdf)
    u = rng.uniform(c_lo, c_hi, size=n)
    return np.interp(u, cdf, xs)
    
'''TOY PLOT''' 
COMPONENTS = ["bkg_Dp", "bkg_Ds", "bkg_Dseta", "bkg_combinatorial"]   # stacking order, bottom -> top
COLORS = {
    "bkg_Dp": "#e0a030",
    "bkg_Ds": "#5aa05a",
    "bkg_Dseta": "#c0504d",
    "bkg_combinatorial": "#4f81bd",
    "sig": "#7f3f98"
}
LEGEND = {
    "bkg_Dp": r"$D^{+}\to3\pi$ misID",
    "bkg_Ds": r"$D_{s}^{+}\to3\pi$ misID",
    "bkg_Dseta": r"$D_{s}^{+}\to\eta\mu\nu$",
    "bkg_combinatorial": "Combinatorial",
    "sig": r"$\tau\to3\mu$"
}
XLABEL = r"$m(\mu^{+}\mu^{-}\mu^{+})$ [MeV/$c^{2}$]"

def _ylabel(bin_width):
    return rf"Candidates / ({bin_width:.3g} MeV/$c^{{2}}$)"

def _stacked(ax, x, label, mask, bins, components, headroom=1.15):
    # stacked histogram of the components + Poisson error bars on the total
    counts = [np.histogram(x[mask & (label == comp)], bins=bins)[0] for comp in components]
    hep.histplot(
        counts, bins=bins, stack=True, histtype="fill", ax=ax,
        color=[COLORS[c] for c in components],
        label=[LEGEND[c] for c in components],
        edgecolor="black", linewidth=0.4
    )
    total = np.sum(counts, axis=0)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.errorbar(centers, total, yerr=np.sqrt(total), fmt="k.", ms=3, lw=1)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, headroom * max(1.0, np.max(total + np.sqrt(total))))

def _shared_legend(fig, ax, ncol, title=None, fontsize=16):
    # one legend for the whole figure, just above the top edge of the axes 'ax'
    handles, labels = ax.get_legend_handles_labels()
    y_top = ax.get_position().y1
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, title=title,
               bbox_to_anchor=(0.5, y_top + 0.005), bbox_transform=fig.transFigure,
               fontsize=fontsize, title_fontsize=fontsize)

def _category_grid(x, label, cbin, base, bins, components, tag, filename, output_dir):
    # 4x4 grid, one panel per bin (category); 'base' selects the years
    with plt.rc_context({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
        axes = axes.flatten()
        first = None
        for c, ax in enumerate(axes):
            mask = base & (cbin == c)
            if not mask.any():
                ax.set_axis_off()          # category not generated (bin 0)
                continue
            _stacked(ax, x, label, mask, bins, components)
            ax.text(0.96, 0.94, f"bin {c}", transform=ax.transAxes, ha="right", va="top", fontsize=12)
            first = ax if first is None else first
        for ax in axes[12:]:
            ax.set_xlabel(XLABEL, fontsize=12)
        for ax in axes[4::4]:
            ax.set_ylabel(_ylabel(bins[1] - bins[0]), fontsize=11)
        fig.tight_layout()
        _shared_legend(fig, first, ncol=len(components), title=f"Toy, {tag}", fontsize=14)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)

def plot_toy(x, cat, label, include_signal=False, output_dir="toy_plots"):
    '''
    12 files (2 regions x 6 figures):
      total, 3 years | total, one panel per year | categories, 3 years | categories, one figure per year
    '''
    os.makedirs(output_dir, exist_ok=True)
    components = COMPONENTS + (["sig"] if include_signal else [])
    year  = cat // p.CATEGORIES
    cbin  = cat % p.CATEGORIES
    everything = np.ones(x.size, dtype=bool)
    regions = {"window": (p.MASS_MIN, p.MASS_MAX), "SR": (p.SR_MIN, p.SR_MAX)}

    with plt.style.context(hep.style.LHCb2):
        for rname, (a, b) in regions.items():
            bins = np.linspace(a, b, int(round((b - a) / p.BIN_WIDTH)) + 1)

            # total, three years together
            fig, ax = plt.subplots(figsize=(10, 7))
            _stacked(ax, x, label, everything, bins, components)
            ax.set_xlabel(XLABEL)
            ax.set_ylabel(_ylabel(p.BIN_WIDTH))
            _shared_legend(fig, ax, ncol=2, title="Toy, 2016-2018")
            fig.savefig(os.path.join(output_dir, f"toy_{rname}_3years.pdf"), bbox_inches="tight")
            plt.close(fig)

            # total, one panel per year
            fig, axes = plt.subplots(1, len(p.YEARS), figsize=(7 * len(p.YEARS), 7), sharey=True)
            for iy, ax in enumerate(axes):
                _stacked(ax, x, label, year == iy, bins, components)
                ax.text(0.05, 0.95, p.YEARS[iy], transform=ax.transAxes, ha="left", va="top", fontsize=18)
                ax.set_xlabel(XLABEL)
            axes[0].set_ylabel(_ylabel(p.BIN_WIDTH))
            _shared_legend(fig, axes[0], ncol=len(components), title="Toy")
            fig.savefig(os.path.join(output_dir, f"toy_{rname}_per_year.pdf"), bbox_inches="tight")
            plt.close(fig)

            # categories: in the full window the bins are twice as wide (few events per bin)
            bw = 2 * p.BIN_WIDTH if rname == "window" else p.BIN_WIDTH
            bins_cat = np.linspace(a, b, int(round((b - a) / bw)) + 1)

            # categories, three years together
            _category_grid(x, label, cbin, everything, bins_cat, components, "2016-2018",
                           f"toy_{rname}_categories_3years.pdf", output_dir)
            # categories, one figure per year
            for iy in range(len(p.YEARS)):
                _category_grid(x, label, cbin, year == iy, bins_cat, components, p.YEARS[iy],
                               f"toy_{rname}_categories_{p.YEARS[iy]}.pdf", output_dir)
                               
'''MODELS BINNING '''
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
