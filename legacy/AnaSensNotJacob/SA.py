# %%
import module as m 
from tkinter import Tk,messagebox
from tkinter.filedialog import askdirectory
from pathlib import Path
import pandas as pd
root = Tk()
root.withdraw()
folderpath = Path(askdirectory(title='Select the SA folder'))
root.destroy()

root = Tk()
root.withdraw()
multielectrode = messagebox.askyesno("Question","Is there more than one electrode saved on each of your file?")
root.destroy()


if multielectrode:
    results = m.analyserodSAfolder(folderpath)
    print(results)
else:
    in_queue = list(folderpath.glob('*.pssession'))

    results = pd.Series(index= range(1,len(in_queue)+1),name = 'Electrochemical Surface Area (x10-3 cm2)')

    for i,f in enumerate(in_queue):
        if i != len(in_queue) -1:
            results.loc[int(f.name.split("_")[0].lstrip("E"))] = m.getSA(f) * 1e3
        else:
            results.loc[int(f.name.split("_")[0].lstrip("E"))] = m.getSA(f,block = True) * 1e3

    print(results)











"""
Work to deal with rod CV measurments
# %%
folderpath = Path(r"E:\stage2020\12rod_20200715\Surface Area")
file_list = list(folderpath.glob("*.pssession"))

# %%
f1 = file_list[0]
V,I,info = m.read_multicurvespssesion(f1)
# %%
for v,i in zip(V,I):
    plt.plot(v,i)
    plt.show()

# %%
def analyserodSAfolder(folderpath,i = -1):
    folderpath = Path(folderpath)
    file_list = list(folderpath.glob("*.pssession"))
    allelist = [int(e) for e  in "".join([f.name.split("_")[0] for f in file_list]).split("E")[1:]]
    results = pd.Series(index = allelist, name = "SAx10_3cm2")
    for f in file_list:
        V,I,info = m.read_multicurvespssesion(f)
        elist = [int(s) for s in f.name.split("_")[0].split("E")[1:]]
        n_electrode = len(elist)
        mespare = int(len(V)/n_electrode)
        for v,c,e in zip([V[k*mespare:(k+1)*mespare] for k in range(n_electrode)],
                         [I[k*mespare:(k+1)*mespare] for k in range(n_electrode)],
                         elist):
            v_,c_ = np.array(v[i]),np.array(c[i])
            k = np.min(np.argwhere(v_[1:] <= v_[:-1])) + 1

            rv,rc = v_[k:],c_[k:]
            ai = np.argmin(np.abs(rv - a))
            bi = np.argmin(np.abs(rv - b))
            rv,rc = rv[ai:bi],rc[ai:bi]


            smooth = signal.savgol_filter(rc,5,1)

            min1,prop = signal.find_peaks(-smooth,
                                            height=None,
                                            threshold=None,
                                            prominence =None,
                                            width = [len(smooth)/20,len(smooth)/1.3])

            l = np.argmax(prop['prominences'])
            max1, max2, min1 = prop['left_bases'][l],prop['right_bases'][l],min1[l]
            auc = get_AUC(rv,smooth,max1,max2)
            SA = (auc/info['scan_rate']/400)*1e3
            results.loc[e] = SA
            if e != max(allelist):
                m.plotall(rv,ri,smooth,max1,min1,max2,f'{filepath.name}',block = False)
            else:
                m.plotall(rv,ri,smooth,max1,min1,max2,f'{filepath.name}',block = True)
    return results
  
"""

            

