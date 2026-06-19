# %%
import module as m
from pathlib import Path
from tkinter import Tk,messagebox
from tkinter.filedialog import askdirectory
import warnings
warnings.filterwarnings('ignore', message = 'Polyfit may be poorly conditioned')


root = Tk()
root.withdraw()
folderpath = Path(askdirectory(title='Select the SA folder'))
root.destroy()

results = m.analyserodMBfolder(folderpath)
print(results)
