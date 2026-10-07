import numpy as np
import pyhf
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyhf.contrib.viz import brazil
from utils import exp_integral, gauss_integral, binning
from utils import get_expected_exp_binned, get_expected_gauss_binned, get_expected_flat_binned

import parameters as p
import toy_generator as tg

pyhf.set_backend("numpy", "minuit")


