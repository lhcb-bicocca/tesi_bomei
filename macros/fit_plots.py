'''
fit_plots.py

Plots of fit result.
Two styles:
- "bins": stacked histogram of the fitted components, bin by bin (as the toy plots)
- "pdf" : smooth curves (total + each component), from the same model built with finer bins and evaluated at the SAME parameters

Output in plots/fit_plots/ (PDF plots) and data/ (text yield tables):
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

import mypkg.parameters as p
from mypkg.utils import COLORS, LEGEND, COMPONENTS, XLABEL, _ylabel, _shared_legend

NAMES = {"signal": "sig", "background_combinatorial": "bkg_combinatorial"} 
FINE_FACTOR = 10 # fine bins per model bin (pdf style)
DATA_DIR = "data"
PLOTS_DIR = os.path.join("plots", "fit_plots")

'''FIT RESULT -> COUNTS'''
def postfit(model, pars):
    # Expected counts of every component at parameters 'pars'
    by_sample = np.asarray(model.main_model.expected_data(pars, return_by_sample = True)) 
    out, start = {}, 0
    for name in model.config.channels:
        n_bins = model.config.channel_nbins[name]
        _, y, _, c = name.split("_")
        out[(y, int(c))] = {NAMES.get(s, s): by_sample[i, start:start + n_bins] for i, s in enumerate(model.config.samples)}
        start += n_bins
    return out

def observed(model, data):
    # Counts used in the fit
    out, start = {}, 0
    for name in model.config.channels:
        n_bins = model.config.channel_nbins[name]
        _, y, _, c = name.split("_")
        out[(y, int(c))] = {"data": np.asarray(data)[start:start + n_bins]}
        start += n_bins
    return out

def fine_model(model, **build_kwargs):
    # Same model with FINE_FACTOR times more bins. 
    # It is only used to draw the fitted pdf as a smooth curve.
    from models import build_workspace
    n_bins = model.config.channel_nbins[model.config.channels[0]]
    old_bin_width = p.BIN_WIDTH
    p.BIN_WIDTH = (p.MASS_MAX - p.MASS_MIN) / (n_bins * FINE_FACTOR)
    try:
        empty = (np.empty(0), np.empty(0, dtype = int), np.empty(0, dtype = "U1"))   
        fine = build_workspace(*empty, **build_kwargs).model()
    finally:
        p.BIN_WIDTH = old_bin_width
    if fine.config.par_order != model.config.par_order:
        raise ValueError("fine model has different parameters: use the same build_kwargs of the fitted model")
    return fine

def yield_table(model, pars, data, log_print_func = print):
    # Fitted yield of each component (all categories) vs nominal, and data.
    init = np.array(model.config.suggested_init(), dtype = float)
    init[model.config.poi_index] = 0.0
    fit, nom = postfit(model, pars), postfit(model, init)
    log_print_func(f"\n{'component':<20} | {'fitted':>10} | {'nominal (mu = 0)':>16}")
    log_print_func("-" * 52)
    for k in COMPONENTS + ["sig"]:
        f, n = sum(v[k].sum() for v in fit.values()), sum(v[k].sum() for v in nom.values())
        log_print_func(f"{k:<20} | {f:10.1f} | {n:16.1f}")
    log_print_func("-" * 52)
    log_print_func(f"{'data':<20} | {sum(v['data'].sum() for v in observed(model, data).values()):10.0f} |")
    log_print_func(f"mu fitted = {pars[model.config.poi_index]:.3f}\n")

'''PLOT'''
def _collect(per_channel, keep):
    # sum over the channels
    tot = {}
    for (y, c), comps in per_channel.items():
        if keep(y, c):
            for k, v in comps.items():
                tot[k] = tot.get(k, 0) + v
    return tot

def _draw(ax, edges, obs, comps, fine_edges, fine_comps, style, headroom = 1.15):
    names = [k for k in COMPONENTS + ["sig"] if k in comps and comps[k].sum() > 0]
    total = sum(comps[k] for k in names)
    if style == "bins":
        hep.histplot(
            [comps[k] for k in names], bins = edges, stack = True, histtype = "fill", ax = ax,
            color = [COLORS[k] for k in names], label = [LEGEND[k] for k in names],
            edgecolor = "black", linewidth = 0.4
        )
    else:
        fine_centers = 0.5 * (fine_edges[1:] + fine_edges[:-1])
        for k in names:
            ax.plot(fine_centers, FINE_FACTOR * fine_comps[k], color = COLORS[k], ls = "--", lw = 1.5, label = LEGEND[k])
        ax.plot(fine_centers, FINE_FACTOR * sum(fine_comps[k] for k in names), color = "black", lw = 2, label = "Fit")
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(centers, obs, yerr = np.sqrt(obs), fmt = "k.", ms = 4, lw = 1, label = "Data", zorder = 5)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, headroom * max(1.0, np.max(np.maximum(obs + np.sqrt(obs), total))))

def plot_fit(model, data, pars, style = "pdf", fine = None, output_dir = PLOTS_DIR, tag = "Fit", ext = "pdf"):
    if style == "pdf" and fine is None:
        raise ValueError("style 'pdf' needs fine = fine_model(...)")
    os.makedirs(output_dir, exist_ok = True)
    n_bins = model.config.channel_nbins[model.config.channels[0]]
    edges = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins + 1)
    fine_edges = np.linspace(p.MASS_MIN, p.MASS_MAX, n_bins * FINE_FACTOR + 1)
    centers, fine_centers = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (fine_edges[1:] + fine_edges[:-1])

    obs_dict, fit_dict = observed(model, data), postfit(model, pars)
    fine_fit_dict = postfit(fine, pars) if style == "pdf" else None
    in_sr = (centers >= p.SR_MIN) & (centers <= p.SR_MAX)
    width = edges[1] - edges[0]

    def view(keep, sel, fsel, e, fe):
        # (counts, components, fine components) of the selected channels in the region; None if there are none
        obs_collected = _collect(obs_dict, keep)
        if not obs_collected:
            return None
        comps = {k: v[sel] for k, v in _collect(fit_dict, keep).items()}
        fcomps = {k: v[fsel] for k, v in _collect(fine_fit_dict, keep).items()} if fine_fit_dict else None
        return e, obs_collected["data"][sel], comps, fe, fcomps

    def legend(fig, ax, title, fontsize = 16):
        _shared_legend(fig, ax, ncol = len(ax.get_legend_handles_labels()[0]), title = title, fontsize = fontsize)

    with plt.style.context(hep.style.LHCb2):
        for rname, region in {"window": np.ones(n_bins, dtype = bool), "SR": in_sr}.items():
            lo, hi = edges[:-1][region][0], edges[1:][region][-1]
            e = np.append(edges[:-1][region], hi)
            fsel = (fine_centers > lo) & (fine_centers < hi)
            fe = np.append(fine_edges[:-1][fsel], fine_edges[1:][fsel][-1])
            v = lambda keep: view(keep, region, fsel, e, fe)

            # total, three years together
            fig, ax = plt.subplots(figsize = (10, 7))
            _draw(ax, *v(lambda y, c: True), style)
            ax.set_xlabel(XLABEL)
            ax.set_ylabel(_ylabel(width))
            legend(fig, ax, f"{tag}, 2016-2018")
            fig.savefig(os.path.join(output_dir, f"fit_{style}_{rname}_3years.{ext}"), bbox_inches = "tight")
            plt.close(fig)

            # total, one panel per year
            fig, axes = plt.subplots(1, len(p.YEARS), figsize = (7 * len(p.YEARS), 7), sharey = True)
            for iy, ax in enumerate(axes):
                _draw(ax, *v(lambda y, c: y == p.YEARS[iy]), style)
                ax.text(0.05, 0.95, p.YEARS[iy], transform = ax.transAxes, ha = "left", va = "top", fontsize = 18)
                ax.set_xlabel(XLABEL)
            axes[0].set_ylabel(_ylabel(width))
            legend(fig, axes[0], tag)
            fig.savefig(os.path.join(output_dir, f"fit_{style}_{rname}_per_year.{ext}"), bbox_inches = "tight")
            plt.close(fig)

            # categories: three years together, then one figure per year
            for ytag, years in [("3years", p.YEARS)] + [(y, [y]) for y in p.YEARS]:
                with plt.rc_context({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10}):
                    fig, axes = plt.subplots(4, 4, figsize = (16, 13), sharex = True)
                    axes = axes.flatten()
                    first = None
                    for c, ax in enumerate(axes):
                        panel = v(lambda y, cc: cc == c and y in years)
                        if panel is None:
                            ax.set_axis_off()
                            continue
                        _draw(ax, *panel, style)
                        ax.text(0.96, 0.94, f"bin {c}", transform = ax.transAxes, ha = "right", va = "top", fontsize = 12)
                        first = ax if first is None else first
                    for ax in axes[12:]:
                        ax.set_xlabel(XLABEL, fontsize = 12)
                    for ax in axes[::4]:
                        ax.set_ylabel(_ylabel(width), fontsize = 11)
                    fig.tight_layout()
                    legend(fig, first, f"{tag}, {'2016-2018' if ytag == '3years' else ytag}", fontsize = 14)
                    fig.savefig(os.path.join(output_dir, f"fit_{style}_{rname}_categories_{ytag}.{ext}"), bbox_inches = "tight")
                    plt.close(fig)

if __name__ == "__main__":
    import mypkg.fast_limit as fl
    from mypkg.toy_generator import generate_toy
    from models import build_workspace

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else p.SEED
    fit_type = sys.argv[2] if len(sys.argv) > 2 else "free"
    styles = ["bins", "pdf"] if (sys.argv[3] if len(sys.argv) > 3 else "both") == "both" else [sys.argv[3]]

    os.makedirs(DATA_DIR, exist_ok = True)
    log_filepath = os.path.join(DATA_DIR, f"fit_yields_seed{seed}_{fit_type}.txt")
    log_file = open(log_filepath, "w", encoding = "utf-8")

    def dual_print(text = ""):
        print(text, flush = True)
        log_file.write(text + "\n")
        log_file.flush()
    
    kw = dict(model_type = p.MODEL, exclude_cat = p.EXCLUDE_CAT)
    x, cat, label = generate_toy(seed, exclude_cat = p.EXCLUDE_CAT, verbose = False)
    model = build_workspace(x, cat, label, **kw).model()
    data = fl.counts(model, x, cat)
    ctx = fl.setup(model)

    nll, pars = fl.fit(ctx, data)
    if fit_type == "bkg":
        nll, pars = fl.fit(ctx, data, mu = 0.0, init = pars)
    dual_print(f"Toy seed {seed}, fit '{fit_type}': -2 ln L = {nll:.2f}")
    yield_table(model, pars, data, log_print_func = dual_print)

    fine = fine_model(model, **kw) if "pdf" in styles else None
    for style in styles:
        plot_fit(model, data, pars, style = style, fine = fine, tag = f"Toy {seed}, {fit_type}")
        
    dual_print(f"Report saved in: {log_filepath}")
    dual_print(f"Plots saved in: {PLOTS_DIR}/")
    
    log_file.close()
