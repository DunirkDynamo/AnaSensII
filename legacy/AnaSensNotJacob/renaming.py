# %%
import os
from pathlib import Path
import re
# %%
folder_path = Path(r"C:\Users\test\Documents\Tanya\15-02-2021 CRT apt 81 - Copy")
pslist = list(folder_path.glob("*.pssession"))
csvlist = list(folder_path.glob("*.csv"))
# %%
for filelist in [pslist,csvlist]:
    for f in filelist:
        unchanged1 = '_'.join(f.name.split("_")[1:])
       # unchanged2= '_'.join(f.name.split("_")[:1])

        if (f.name.split("_")[0] == "E1"): 
            f.rename(f.parent / ("E1_hybridization_" +  unchanged1))


# %%
