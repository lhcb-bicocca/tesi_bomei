import parameters as p
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil

def upper_limit(workspace, level = p.LEVEL, PLOT = False, VERBOSE = True):
    model = workspace.model()
    data  = workspace.data(model)

    ul_obs_grid, ul_exp_grid = pyhf.infer.intervals.upper_limits.upper_limit(data, model, np.linspace(0,50,100), level=level)
    
    BR_obs_grid = ul_obs_grid * p.BR_REF
    BR_exp_grid = np.asarray(ul_exp_grid) * p.BR_REF
    
    if VERBOSE:
	    print(f"UPPER LIMIT CON ASINTOTICO E GRID")
	    print(f"BR_obs (90% CL) = {BR_obs_grid}")
	    print("BR expected (90% CL):")
	    print(f"  median         = {BR_exp_grid[2]}")
	    print(f"  -2σ            = {BR_exp_grid[0]}")
	    print(f"  -1σ            = {BR_exp_grid[1]}")
	    print(f"  +1σ            = {BR_exp_grid[3]}")
	    print(f"  +2σ            = {BR_exp_grid[4]}")
	    print("\n")

	#Asymptotic with rootfinder (toms748)
	
    ul_obs_root, ul_exp_root = pyhf.infer.intervals.upper_limits.upper_limit(data, model, level=level)
    
    BR_obs_root = ul_obs_root * p.BR_REF
    BR_exp_root = np.asarray(ul_exp_root) * p.BR_REF

    if VERBOSE:
	    print(f"UPPER LIMIT CON ASINTOTICO E TOMS748")
	    print(f"BR_obs (90% CL) = {BR_obs_root}")
	    print("BR expected (90% CL):")
	    print(f"  median         = {BR_exp_root[2]}")
	    print(f"  -2σ            = {BR_exp_root[0]}")
	    print(f"  -1σ            = {BR_exp_root[1]}")
	    print(f"  +1σ            = {BR_exp_root[3]}")
	    print(f"  +2σ            = {BR_exp_root[4]}")
	    print("\n")
	
	
    if PLOT:
	    #brazil plot
	    test_mus = np.linspace(0, 10, 100)
	    results = [pyhf.infer.hypotest(m, data, model, test_stat="qtilde", return_expected_set=True) for m in test_mus]

	    fig, ax = plt.subplots()
	    brazil.plot_results(test_mus*p.BR_REF, results, test_size = level, ax=ax)

	    ax.set_xlabel("Br", fontsize=12)
	    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	    ax.set_ylim(0, 1.0)

	    plt.show()
	    
    return BR_obs_grid, BR_exp_grid, BR_obs_root, BR_exp_root
