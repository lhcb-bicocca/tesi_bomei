from analysis_helpers.plotting import plot_hist2d
from numpy.random import rand
import matplotlib.pyplot as plt


array1 = 10*rand(1000)
array2 = 10*rand(1000)

plot_hist2d(array1, array2)

plt.show()
