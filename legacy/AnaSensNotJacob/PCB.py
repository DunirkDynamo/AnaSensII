# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import module as m
from pathlib import Path
import matplotlib.patches as mpatches

# %%
#loading the results csv
filepath = Path(r"D:\Simon\New_PCB\Lot111\Hybridization_100Hz\\analysis\\analysis.csv")


results = pd.read_csv(filepath,index_col=0)


results["time"] = pd.to_datetime(results["time"])



results = pd.concat([results]).reset_index(drop = True)
#results = results.loc[results["fit_score"] >= 0.95,:].reset_index(drop = True)
n_electrode = 16

# %%
#results["nth_use"] = [int(f.split("_")[2]) for f in results["filename"]]
#results["concentration"] = [int(f.split("-")[0].split("_")[-1]) for f in results["filename"]]
results["time"] = pd.to_datetime(results["time"])
results = results.loc[results["fit_score"] >= 0.9,:]

# %%  PH 20 minutes
fig,ax = plt.subplots()

features = m.features_extraction(results,from_path = False)
concentration_list = sorted(list(set(features.concentration)))

for i,c in enumerate(concentration_list):
    sub = features.loc[features.concentration == c,:]
    mean_ph20m = sub["ph20m"].mean()
    std_ph20m = sub["ph20m"].std()
    rel_std = 100*std_ph20m/mean_ph20m


    ax.bar([i,],
        height = [mean_ph20m,],
        yerr = [std_ph20m,],
        capsize = 5)
ax.grid()
plt.xticks(range(len(concentration_list)))
ax.set_title("Peak Height(nA) at 20m for different \n analyte concentration")
ax.set_xlabel("Analyte concentration")
ax.set_ylabel("peak height at 20m (nA)")
ax.set_xticklabels(concentration_list)
fig.savefig(Path(r"D:\Simon\New_PCB\Lot111\Hybridization_100Hz\\analysis\\rodsc0c1_barplot.png"))

# %%  PH 10 minutes
fig,ax = plt.subplots()

features = m.features_extraction(results,from_path = False)
concentration_list = sorted(list(set(features.concentration)))

for i,c in enumerate(concentration_list):
    sub = features.loc[features.concentration == c,:]
    mean_ph10m = sub["ph10m"].mean()
    std_ph10m = sub["ph10m"].std()
    rel_std = 100*std_ph10m/mean_ph10m


    ax.bar([i,],
        height = [mean_ph10m,],
        yerr = [std_ph10m,],
        capsize = 5)
ax.grid()
plt.xticks(range(len(concentration_list)))
ax.set_title("Peak Height(nA) at 10m for different \n analyte concentration")
ax.set_xlabel("Analyte concentration")
ax.set_ylabel("peak height at 10m (nA)")
ax.set_xticklabels(concentration_list)
fig.savefig(Path(r"D:\Simon\New_PCB\Lot111\Hybridization_100Hz\\analysis\\rodsc0c1_barplot.png"))


# %%  PH 5 minutes
fig,ax = plt.subplots()

features = m.features_extraction(results,from_path = False)
concentration_list = sorted(list(set(features.concentration)))

for i,c in enumerate(concentration_list):
    sub = features.loc[features.concentration == c,:]
    mean_ph10m = sub["ph5m"].mean()
    std_ph10m = sub["ph5m"].std()
    rel_std = 100*std_ph10m/mean_ph10m


    ax.bar([i,],
        height = [mean_ph10m,],
        yerr = [std_ph10m,],
        capsize = 5)
ax.grid()
plt.xticks(range(len(concentration_list)))
ax.set_title("Peak Height(nA) at 5m for different \n analyte concentration")
ax.set_xlabel("Analyte concentration")
ax.set_ylabel("peak height at 5m (nA)")
ax.set_xticklabels(concentration_list)
fig.savefig(Path(r"D:\Simon\New_PCB\Lot111\Hybridization_100Hz\\analysis\\rodsc0c1_barplot.png"))


# %%
subb = features.loc[[1,2,3,4,6,7,8,9,10,11,12,13,14,15],["concentration","ph20m"]].astype(np.float32)
a = subb.groupby("concentration").mean()

b =subb.groupby("concentration").std()
c = (b/a)*100

gb = subb.groupby("concentration")

# %%
import numpy as np
subb = features.loc[:,["ph5m","ph10m","ph20m"]].astype(np.float32)
a = subb.mean()
b = subb.std()
c = (100*subb.std()/subb.mean())
variation = [a,b,c]

print ("\n        Mean\n\n",a)

print ("\n      STD\n\n",b)

print ("\n      Error %\n\n",c)


# %%

import numpy as np
subb = features.loc[:,["ph5m","ph10m","ph20m"]].astype(np.float32)
a = subb.mean().tolist()
b = subb.std().tolist()
c = (100*subb.std()/subb.mean()).tolist()
variation = [a,b,c]
print ("Results\n",a)

print ("Results\n",b)

print ("Results\n",c)


# 5min
mean_ph5m = subb["ph5m"].mean()
std_ph5m = subb["ph5m"].std()
rel_std5 = 100*std_ph5m/mean_ph5m
# 10min
mean_ph10m = subb["ph10m"].mean()
std_ph10m = subb["ph10m"].std()
rel_std10 = 100*std_ph10m/mean_ph10m
# 5min
mean_ph20m = subb["ph20m"].mean()
std_ph20m = subb["ph20m"].std()
rel_std20 = 100*std_ph20m/mean_ph20m



# %%
