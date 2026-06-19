#%%
import numpy as np 
import pandas as pd 
import os 
#%%
filepath = r"F:\Simon\Ageing\Vacuum Effect\Day1\Oxygen\Hybrid\analysis\analysis.csv"
results = pd.read_csv(filepath,index_col=0)
results["time"] = pd.to_datetime(results["time"])
#%%
sub = results.loc[:,["time","electrode","PH"]]
#results.loc[(results["electrode"] == 21 & results["time"] < 15)]
list(results.loc[(results["electrode"]==21) & (results["nth_use"] ==0),"PH"])
#%%
results = results.sort_values(["electrode","time"])
#results3 =