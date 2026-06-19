
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import module as m
from pathlib import Path
import matplotlib.patches as mpatches
from collections import OrderedDict



# %%
#loading the results csv
filepath = Path(r"D:\\Simon\\PCB-BR\\Test new PCB\\analysis\\analysis.csv")
#D:\Simon\New_PCB\Lot111\Hybridization_100Hz\\analysis\\analysis.csv
#D:\Simon\PCB-BR\100 Frequency\analysis
#D:\Simon\PCB-BR\Test new PCB\analysis

results = pd.read_csv(filepath,index_col=0)


results["time"] = pd.to_datetime(results["time"])



results = pd.concat([results]).reset_index(drop = True)
#results = results.loc[results["fit_score"] >= 0.95,:].reset_index(drop = True)
n_electrode = 16
graphics = True
legend_location = 'best'
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
            #if f<=8:
            if f in [1,2] :
                 color = 'black'
                 #my_alpha = 1 - (f-1)/9
            #else : 
            if f in [3,4]:
                color = 'blue'
                #my_alpha = 1 - (f-9)/9
            if f in [5,6]:
                color = 'grey'
            if f in [7,8]:
                color = 'orange'
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
                
                    #axph.set_xlim(right=10)
                    #axph.set_ylim(top=450)
                    #print(my_alpha)
                    axpeakpos.scatter(time,pp,5,color,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")#, alpha = my_alpha)

                    axbk.scatter(time,bk,5,color,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")#, alpha = my_alpha)

                    axph.scatter(time,ph,5,color,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")#, alpha = my_alpha)

                    axauc.scatter(time,AUC,5,color,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")#, alpha = my_alpha )
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

#fig.savefig(Path(f"D:\\Simon\\mesures_ML\\Rod analysis\\ConcentrationLogPH20m.png"))

figph.savefig(Path(r"D:\\Simon\\PCB-BR\\Test new PCB\\analysis\\PeakHeight2.png"))
figauc.savefig(Path(r"D:\\Simon\\PCB-BR\\Test new PCB\\analysis\\AreaUnderCurve2.png"))
figpeakpos.savefig(Path(r"D:\\Simon\\PCB-BR\\Test new PCB\\analysis\\PeakPosition2.png"))
figbk.savefig(Path(r"D:\\Simon\\PCB-BR\\Test new PCB\\analysis\\AvgBackGround2.png"))

# %%




################################################################################
######################### --- FIGURES START --- ################################
################################################################################
if graphics:
    print('producing graphics...')
    results_ = results.loc[results["fit_score"] >= 0.935,:]
       electrode_set = sorted(list(set(results_["electrode"])))
    concentration_set = sorted(list(set(results_["concentration"])))
    figph,axph = plt.subplots()
    
    compteur = 0
    i = 0


    for c in concentration_set:
        for f in electrode_set: 
            if f<=8: color = 'red'
            else : color = 'black'
            for n in nth_use_set:
                
               # time = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'time']<=10
                time = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'time']
                time = time - time.min()
            
                time = np.divide(time, np.timedelta64(1,'m'))
                ph   = results_.loc[(results_['electrode'] == f) & (results_["concentration"] == c) & (results_["nth_use"] == n),'PH'].values
               
                if len(ph) != 0:
                    ph = ph - ph[0]
                
                
                
                    axph.scatter(time,ph,5,color,marker = symbols[compteur], label = f"E{f} - c : {c} - n : {n}")
                    i += 1
                    if (i+1) % 10 == 0:
                        compteur += 1



    axph.set_xlabel("time (min)")
    axph.set_ylabel(f"Peak height (nA)")
    axph.set_title(f"Peak Height vs time")

    axph.legend(loc = legend_location)   

    axph.grid()
# %%
