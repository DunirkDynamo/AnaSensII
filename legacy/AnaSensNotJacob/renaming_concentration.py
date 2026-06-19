# %%
import os
from pathlib import Path
import re
# %%
folder_path = Path(r"C:\Users\test\Documents\Tanya\26-01-2021 CTI_Moxi") 
pslist = list(folder_path.glob("*.pssession"))
csvlist = list(folder_path.glob("*.csv"))
# %%
for filelist in [pslist,csvlist]:
    for f in filelist:
        unchanged1 = '_'.join(f.name.split("_")[1:])
        #unchanged2 = f.name.split("-")[0]
        if f.name.split("_")[0] == "E6E23E26":
            f.rename(f.parent / ("E1" + "_" + unchanged1))


# %%
