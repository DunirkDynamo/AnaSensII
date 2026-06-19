################################################################################
############################## --- IMPORTS ---##################################
################################################################################
print('importing librairies and reading inputs \n')
import numpy as np 
import pandas as pd 
import module as m 
import matplotlib.pyplot as plt 
from pathlib import Path
import warnings
import os
import argparse
from time import time as t
from tkinter import Tk
from tkinter.filedialog import askdirectory
import re
from sklearn.metrics import r2_score

t1 = t()
################################################################################
############################ --- USER INPUTS --- ###############################
################################################################################
#-------------------------------------------------------------------------------
# NON OPTIONAL
#-------------------------------------------------------------------------------
# folderpath : path to the folder containing the csv files.
#   e.g : folderpath = r'C:\Users\pierr\REPO\stage2020\data'

#folderpath = Path(r'E:\stage2020\6rod_20200729\Hybridization-100Hz')

# file_extension : extension of the files to analyse. Either 'csv' or 'pssession'
file_extension = 'pssession'
#-------------------------------------------------------------------------------
# OPTIONAL
#-------------------------------------------------------------------------------
# graphics : boolean value, will produce graphics if True and won't if False
graphics = True

# sep : separator used in csv creation. Desired value depends on your computer's
#   settings. Should be either ',' or ';'
sep = ','

# cropping : boolean value. If True, the script will ask the user to specify a
#   number of data points to remove on each side of all scans. Crop your data if
#   the potential window is too large or if you have weird mesure at extreme 
#   potential
cropping = True
#-------------------------------------------------------------------------------
# TECHNICAL (even more optional...)
#-------------------------------------------------------------------------------
# savgol_win_len : length of the window used in the Savitzky-Golay filter.
# Has to be an odd integer
savgol_win_len = 9 

# savgol_polyorder : order the polynome fit done on each of the Savitzky-Golay
# window. (Based on what I read, it shouldn't exceed 3...)
savgol_polyorder = 1

# polyfit_order : order of the polynome fit done on the whole data. I found that 
# the higher the order, the better our data was approximated
polyfit_order = 60

# legend_location : position of the legend in the generated graphics.  
# See matplotlib.pyplot documentation for more details:
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
# The allowed values are :
# {'best','upper right','upper left','lower left','lower right','right',
# 'center left', 'center right', 'lower center', 'upper center', 'center'}
legend_location = 'best'

# more in the future...
################################################################################
root = Tk()
root.withdraw()
folderpath = Path(askdirectory(title='Select the folder you want to analyse'))
root.destroy()

################################################################################
################################################################################
################################################################################

################################################################################
######################## --- CODE EXECUTION --- ################################
################################################################################

# argsparser setup 
descr = """Automatically analyse SWV mesures. 
Open this script in a text editor to specify your parameters."""
parser = argparse.ArgumentParser(description= descr)
args = parser.parse_args()

##################### --- user's input verification --- ########################
adm_values = {'best','upper right','upper left','lower left','lower right','right',
    'center left', 'center right', 'lower center', 'upper center', 'center',
    0,1,2,3,4,5,6,7,8,9,10}
err_mes = 'The value you have entered for « legend_location » is not admissible'
assert legend_location in adm_values, err_mes

# We ignore the warning produced by the np.polyfit function...  It seems to find
# the order too high, but I have a lot of visual confirmation that it is a 
# better approximation of our data!
warnings.filterwarnings('ignore', message = 'Polyfit may be poorly conditioned')
################################################################################
# We transform the folderpath string to a pathlib.Path object
#folderpath = Path(folderpath)
assert folderpath.exists(), 'The given path is invalid'

# make sure savgol_win_len is odd
assert savgol_win_len % 2 == 1, 'Make sure savgol_win_len is an odd number'

assert file_extension in {'csv','pssession'}, f'{file_extension} file format not supported'
################################################################################
#We chose the appropriate reading function
if file_extension == 'csv':
    reading_func = m.read_pscsv
elif file_extension == 'pssession':
    reading_func = m.read_pssession
# We place all the files to analyse in a list
in_queue = list(folderpath.glob(f"*.{file_extension}"))

# We add '-0' to the first pssession files of each experiments if needed
to_correct = [f for f in in_queue if re.search(f"(-[0-9]*\.{file_extension})$",f.name) == None]
if to_correct:
    for f in to_correct:
        f.rename(f.parent / (f.stem+'-0'+f.suffix))
    in_queue = list(folderpath.glob(f"*.{file_extension}"))
 
# We sort the list using natural sorting functions found online
in_queue = sorted(in_queue, key = lambda p : m.natural_keys(p.name))

# We read on of these file and ask the user how many data points we should
# ignore on each sides (allows to clean outliers that sometime happen at the
# extreme potentials...). Now only if specified.
t2 = t()
#Reading the data
V,I,info = m.read_rodfolder(in_queue,samelen = False)

if cropping:
    n_to_drop = m.mptinput_cropping2(V,I)
t3 = t()
# We create a pandas.DataFrame (a kind of array) with predefined dimension
# to optimise effiency
data = pd.DataFrame(index= pd.RangeIndex(0,len(V)),
                     columns = ['PH','AUC','peak_pos','bkgrd_avg_current',"fit_score"]
                     )
print(f"Starting analysis on {len(in_queue)} files ({len(V)} curves)")
# we iterate of every file
for j,v,i in zip(range(len(V)),V,I):
    # isolate the essai name using regular expression
    #name = re.sub(r"_*-*[0-9]*(\([0-9]\))*\."+file_extension,"",file.name)
    
    # isolate the file number
    #number = file.name.split("-")[-1].rstrip(f".{file_extension}")

    # extract data from csv
    #V,I,info = reading_func(file)

    #remove data points initailly specified by the user
    if cropping:
        v, i = v[n_to_drop: -n_to_drop], i[n_to_drop: -n_to_drop]

    # we apply the Savitzky-Golay filter and we fit the data with a polynome
    smoothed_I,coeffs = m.data_smoothing(v,i,savgol_win_len,savgol_polyorder,polyfit_order,coeffs = True)
    fit_score = r2_score(i,smoothed_I)
    #evaluating derivative
    d_c = np.polyder(coeffs)
    d = np.polyval(d_c,v)
    
    # We find the peak position using sp.signal
    try:
        max1,max2,ipeak = m.findpeaksp(v,smoothed_I,d)
    except:
        max1,max2,ipeak = [0,0,0]
    # We find the peak position in Volt
    if ipeak != 0:
        Vpeak = v[ipeak]
        # We get the peak heigth, the area under the peak and the average background current
        peak_heigth = m.get_peakheight(v,smoothed_I,max1,max2,ipeak)
        auc = m.get_AUC(v,smoothed_I,max1,max2)
        bkgrd_avg_current = m.get_bg_avg(v,i,auc)
        # We fill our array
        data.loc[j,:] = [peak_heigth,auc,Vpeak,bkgrd_avg_current,fit_score]
    else:
        Vpeak = np.nan
        peak_heigth = 0
        auc = 0
        bkgrd_avg_current = m.get_bg_avg(v,i,auc)
        data.loc[j,:] = [peak_heigth,auc,Vpeak,bkgrd_avg_current,fit_score]
    if j % 5 == 0:
        print(f"{j} / {len(V)} completed.\r",end = "")

results = info

results["PH"] = data["PH"] * 1e3
results["AUC"] = data["AUC"] * 1e3
results["peak_pos"] = data["peak_pos"]
results["bkgrd_avg_current"] = data["bkgrd_avg_current"] * 1e3
results["fit_score"] = data["fit_score"]

try:
    results["concentration"] = list([int(i.split("_")[-1]) for i in results["label"]])
    results["nth_use"] = list([int(i.split("_")[-2]) for i in results["label"]])
except:
    results["concentration"] = np.nan
    results["nth_use"] = np.nan
    graphics = False
    print("No graphics will be produce : Data format not compatible ")





# We impose appropriate data type on our DataFrame
results["time"] = pd.to_datetime(results["time"])
results = results.sort_values(["filename",'time']).reset_index(drop = True)




################################################################################
######################### --- FIGURES START --- ################################
################################################################################
if graphics:
    print('producing graphics...')
    results_ = results.loc[results["fit_score"] >= 0.935,:]
    electrode_set = sorted(list(set(results_["electrode"])))
    concentration_set = sorted(list(set(results_["concentration"])))
    nth_use_set = sorted(list(set(results_["nth_use"])))
    figph,axph = plt.subplots()
    figauc,axauc = plt.subplots()
    figpeakpos,axpeakpos = plt.subplots()
    figbk,axbk = plt.subplots()
    symbols = [".","^","o","x"]
    compteur = 0
    i = 0


    for c in concentration_set:
        for f in electrode_set:
            for n in nth_use_set:
            
                time = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'time']
                time = time - time.min()
                time = np.divide(time, np.timedelta64(1,'m'))
                ph   = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'PH'].values
                AUC  = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'AUC'].values
                pp   = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'peak_pos']
                bk   = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n) ,'bkgrd_avg_current']
                if len(ph) != 0:
                    ph = ph - ph[0]
                    AUC = AUC - AUC[0]
                
                
                    axpeakpos.scatter(time,pp,5,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")

                    axbk.scatter(time,bk,5,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")

                    axph.scatter(time,ph,5,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")

                    axauc.scatter(time,AUC,5,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}" )
                    i += 1
                    if (i+1) % 10 == 0:
                        compteur += 1


    axpeakpos.set_xlabel("time (min)")
    axpeakpos.set_ylabel("Potential (V)")
    axpeakpos.set_title("Peak position vs time")

    axbk.set_xlabel("time (min)")
    axbk.set_ylabel(f"Average Background (nA)")
    axbk.set_title("Average background vs time")

    axph.set_xlabel("time (min)")
    axph.set_ylabel(f"Peak height (nA)")
    axph.set_title(f"Peak Height vs time")

    axauc.set_xlabel("time (min)")
    axauc.set_ylabel(f"AUC (nA)")
    axauc.set_title(f"Area under curve vs time")

    axph.legend(loc = legend_location)   
    axauc.legend(loc = legend_location) 
    axpeakpos.legend(loc = legend_location)
    axbk.legend(loc = legend_location)

    axph.grid()
    axauc.grid()
    axpeakpos.grid()
    axbk.grid()


################################################################################
########################### --- FIGURES STOP --- ###############################
################################################################################
# We save our result as a csv and png file
# if the location does not exixts, we create it before saving
if not os.path.exists(folderpath / "analysis"):
    os.makedirs(folderpath / "analysis")
    
results = results.sort_values(["electrode","time"])
results.to_csv(folderpath / "analysis" / "analysis.csv", sep = sep)

if graphics:
    figph.savefig(folderpath / "analysis" / "PeakHeight.png")
    figauc.savefig(folderpath / "analysis" / "AreaUnderCurve.png")
    figpeakpos.savefig(folderpath / "analysis" / "PeakPosition.png")
    figbk.savefig(folderpath / "analysis" / "AvgBackGround.png")
    
print(f"Done! analysed {len(in_queue)} files in {t() - t1 - (t3-t2):.4f} seconds")