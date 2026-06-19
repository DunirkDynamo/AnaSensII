# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict

X = np.linspace(0,100,100)

fig,ax = plt.subplots()
for a in np.arange(0,10):
    ax.plot(X,a*X,'k.',alpha = a/10)

# %%
