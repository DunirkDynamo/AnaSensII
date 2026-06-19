from tkinter import Tk, ttk
import tkinter as tk
from tkinter.filedialog import askdirectory
from pathlib import Path
import os
import pickle
import pandas as pd
from scipy.integrate import simpson
import numpy as np
from scipy.signal import savgol_filter
from pybaselines import Baseline
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import Ekidna as ek
from tkinter import messagebox
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

def select_folder():
    '''
    Purpose: To allow the user to select a folder in which voltammogram data is saved as csv files

    Output:
    - folderpath (str) : The path to a folder containing voltammograms stored as csv files.
    '''

    # Use a tkinter window to navigate to the folder in "point-and-click" fashion:
    root = Tk()
    root.withdraw() # hides the main tkinter window because it's unnecessary for the purpose of the function
    folderpath = askdirectory(title='Select the folder you want to analyse') # creates a window that allows the user to navigate to the folder storing voltammograms as csv files
    root.destroy() # get rid of the tkinter windows 
    return folderpath


def load_voltammograms(folderpath):
    '''
    Purpose: Load all voltammograms saved as csv files in a specific folder. Pair them in a list with their filenames.

    Input: 
    - folderpath (str) : The path to a folder that stores voltammograms as csv files

    Output:
    - voltammogram_list (list) : A list of dict-type data where each dict contains a voltammogram as a pandas dataframe
                                 and its corresponding filename as a string. 
    '''


    filenames = os.listdir(folderpath) # make a list of all files in folder spcified by folderpath
    filenames = [file for file in filenames if file.endswith('.csv')] # drops any non-csv files from the list (we only want to load the csv files..
                                                                      # the user should make sure that only the correct data are in the folder but
                                                                      # this is a second check)
    # loop over files in folderpath, read them into pandas dataframes, and add them to a list
    voltammogram_list      = []
    for file in filenames:
        filepath     = folderpath + '\\' + file # add current file name to folderpath to get file path
        voltammogram = pd.read_csv(filepath)    # read voltammogram/csv into dataframe
        voltammogram_list.append({'voltammogram' : voltammogram, 
                                  'file' : file
                                  })  # creat a dict, containing the dataframe/voltammogram and its filename, to list

   
    return voltammogram_list

def load_single_measurement(folderpath):
    '''
    Purpose: Load all voltammograms saved as csv files in a specific folder. Pair them in a list with their filenames.

    Input: 
    - folderpath (str) : The path to a folder that stores voltammograms as csv files

    Output:
    - voltammogram_list (list) : A list of dict-type data where each dict contains a voltammogram as a pandas dataframe
                                 and its corresponding filename as a string. 
    '''


    filenames = os.listdir(folderpath) # make a list of all files in folder spcified by folderpath
    filenames = [file for file in filenames if file.endswith('.csv')] # drops any non-csv files from the list (we only want to load the csv files..
                                                                      # the user should make sure that only the correct data are in the folder but
                                                                      # this is a second check)
    # loop over files in folderpath, read them into pandas dataframes, and add them to a list
    voltammogram_list      = []
    for file in filenames:
        filepath     = folderpath + '\\' + file # add current file name to folderpath to get file path
        voltammogram = pd.read_csv(filepath)    # read voltammogram/csv into dataframe
        voltammogram_list.append({'voltammogram' : voltammogram, 
                                  'file' : file
                                  })  # creat a dict, containing the dataframe/voltammogram and its filename, to list

   
    return voltammogram_list


def load_many_measurements(parent_folderpath):
    '''
    Output:
    - all_measurement_list (list) : a list of sublists which themselves contain many dictionaries
                                    , each of which contains information for a single sweep of a 
                                    measurement (which is a collection of sweeps).
    '''
    subfolders           = os.listdir(parent_folderpath)
    all_measurement_list = []
    for folderpath in subfolders:
        subfolder_path = parent_folderpath + '\\' + folderpath
        filenames = os.listdir(subfolder_path) # make a list of all files in folder spcified by folderpath
        filenames = [file for file in filenames if file.endswith('.csv')] # drops any non-csv files from the list (we only want to load the csv files..
                                                                        # the user should make sure that only the correct data are in the folder but
                                                                        # this is a second check)
        # loop over files in folderpath, read them into pandas dataframes, and add them to a list
        voltammogram_list = []
        for file in filenames:
            filepath     = subfolder_path + '\\' + file # add current file name to folderpath to get file path
            voltammogram = pd.read_csv(filepath)    # read voltammogram/csv into dataframe
            voltammogram_list.append({'voltammogram' : voltammogram, 
                                    'file' : file
                                    })  # creat a dict, containing the dataframe/voltammogram and its filename, to list
        all_measurement_list.append(voltammogram_list)
   
    return all_measurement_list, subfolders





def choose_cutoff_with_visual_assist(dataframes):
    """
    Purpose:
        This function creates a Tkinter window where the user can visually inspect random plots from the 
        input data and select how many leading samples to cut.

    Input:
        dataframes (list): A list of pandas DataFrames containing columns 'V' and 'I'.

    Output:
        num_cut (int): The number of leading samples the user wants to remove.
    """
    
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Data Plotting Tool")

    # Randomly select up to 4 dataframes
    num_to_plot = min(len(dataframes), 4)
    plot_indices = random.sample(range(len(dataframes)), num_to_plot)
    selected_dataframes = [dataframes[i] for i in plot_indices]

    # Set up matplotlib figure and axes
    fig, axes = plt.subplots(1, num_to_plot, figsize=(20, 10/num_to_plot), sharey=True)
    if num_to_plot == 1:
        axes = [axes]  # Ensure axes is always a list

    # Tkinter canvas for embedding Matplotlib plots
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # Variable to store user input
    num_cut = tk.IntVar(value=0)  # Using IntVar for better tracking
    final_cut = 0  # Will store the final confirmed value

    def create_plot():
        """Plots the selected DataFrames."""
        for ax, df, idx in zip(axes, selected_dataframes, plot_indices):
            ax.clear()
            ax.plot(df['V'], df['I'], label=f"DF {idx}")
            ax.set_xlabel('V')
            ax.set_ylabel('I')
            ax.legend()
            ax.set_title(f"Plot {idx}")
        canvas.draw()

    def update_plot():
        """Updates the plot based on the chosen cut value."""
        try:
            cut_value = int(entry.get())
            if cut_value < 0:
                raise ValueError("Negative value")
                
            num_cut.set(cut_value)  # Update the IntVar
            cut_dataframes = [df.iloc[cut_value:] for df in selected_dataframes]

            # Redraw the plots with the updated data
            for ax, df in zip(axes, cut_dataframes):
                ax.clear()
                ax.plot(df['V'], df['I'])
                ax.set_xlabel('V')
                ax.set_ylabel('I')
            canvas.draw()

            # Ask user for confirmation
            if messagebox.askyesno("Confirm", f"Cut first {cut_value} samples?\nClick No to modify."):
                nonlocal final_cut
                final_cut = cut_value
                root.quit()  # Exit mainloop
                root.destroy()  # Close the window
                
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a positive integer.")

    # Create the initial plot
    create_plot()

    # User input section
    tk.Label(root, text="Enter number of leading samples to cut:", 
             font=('Arial', 16)).pack()
    entry = tk.Entry(root, font=('Arial', 16), width=10)
    entry.pack()
    
    tk.Button(root, text="Apply Cut", command=update_plot, 
              font=('Arial', 16)).pack()

    # Start the Tkinter event loop
    root.mainloop()

    # Clean up matplotlib figures
    plt.close(fig)
    
    return final_cut




def add_voltammogram_copies_to_list(voltammogram_list, voltammograms_processed):
    '''
    Purpose: voltammogram_list is a list of dict objects. Each dict contains a voltammogram. The purpose of this function
             is to add, to each dict, a copy of its existing voltammogram. These copied versions will be altered in the 
             rest of the program, but the original voltammogram will remain unchanged.
    
    Input:
    - voltammogram_list (list)       : a list of dict objects. The relevant element of the dict object is a DataFrame of a voltammogram.
    - voltammograms_processed (list) : a list of DataFrame objects, which are copies of the voltammograms in the dict objects
                                       within voltammogram_list
    
    Effect:
    - Each sub-dict within voltammogram_list has a new element, called 'voltammogram_processed', which is a copy of the existing
      'voltammogram' entry.
    '''
    num_grams = len(voltammogram_list)

    for  i in range(num_grams):
        voltammogram_list[i]['voltammogram_processed'] = voltammograms_processed[i]

    return


def parse_filename(file):
    parts = file.split('_')

    
    
    subparts  = parts[1].split('-')
    Q1        = subparts[0]
    label     = subparts[1]
    timestamp = float(subparts[2][:-1])

    qualifiers = {
                    'Label'     : label,
                    'Timestamp' : timestamp,
                    'Q1'        : Q1,
                    'Date'      : parts[0]
                }

    return qualifiers

def loop_parse_filenames(filenames):
    num_files       = len(filenames)
    qualifiers_list = []
    for i in range(num_files):
        curr_qualifiers = parse_filename(filenames[i])
        qualifiers_list.append(curr_qualifiers)
    return qualifiers_list

def add_qualifiers_to_voltammogram_list(voltammogram_list, qualifier_list):
    '''
    Purpose: Add qualifiers from parsed filenames to the voltammogram_list which contains voltammogram data and a filename.
            The sublists from qualifier_list are added to the sublists of voltammogram_list.
            The 1st sublist of voltammogram_list is merged with the 1st sublist of qualifier_list, the 2nd with
            the 2nd, and so on.
            Each merged sublist contains the voltammogram as a dataframe, the corresponding filename of the voltammogram
            (i.e. filename of original csv file), as well as qualifiers parsed from the filenames.

    Input:
    - voltammogram_info (list)
    - qualifier_list    (list)
    '''
    num_grams = len(voltammogram_list)
    for i in range(num_grams):
        voltammogram_list[i]['qualifiers'] = qualifier_list[i]

    return



def integrate_simpson(x, y, x_min, x_max):
    """
    Purpose: Integrates discrete (x, y) data using Simpson’s rule within given bounds.

    Parameters:
        x (float/array-like) : The x-values (must be sorted in ascending order).
        y (float/array-like) : The corresponding y-values.
        x_min (float)        : Lower bound of integration.
        x_max (float)        : Upper bound of integration.

    Returns:
        area (float)         : Approximate integral value within [x_min, x_max].
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Select data within bounds
    mask = (x >= x_min) & (x <= x_max)
    x_subset = x[x_min:(x_max+1)]
    y_subset = y[x_min:(x_max+1)]

    # Check if we have enough points for Simpson’s rule (needs at least 3 points)
    if len(x_subset) < 3:
        raise ValueError("Simpson’s rule requires at least 3 data points within the bounds.")

    # Ensure an odd number of points (required for Simpson’s rule)
    if len(x_subset) % 2 == 0:
        x_subset = x_subset[:-1]  # Remove the last point to make it odd
        y_subset = y_subset[:-1]

    # Compute the integral using Simpson’s rule
    area = simpson(y_subset, x_subset)
    
    return area

def loop_integrate_peaks(extracted_params, voltammograms_processed):
    num_grams = len(voltammograms_processed)


    for i in range(num_grams):
        num_peaks = len(extracted_params[i]['peak_indices'])
        potential = voltammograms_processed[i]['V'].to_numpy()
        current   = voltammograms_processed[i]['I_baseline_sub'].to_numpy()

        FWHM_integrals = np.zeros(num_peaks)
        FW01_integrals = np.zeros(num_peaks)
        for j in range(num_peaks):

            # Integrate peak over range spanning > 0.5 Max
            left_index        =  extracted_params[i]['left_edge_indices'][j]
            right_index       = extracted_params[i]['right_edge_indices'][j]
            integral          = integrate_simpson(potential, current, left_index, right_index)
            FWHM_integrals[j] = integral

            # Integrate peak over range spanning > 0.1 Max
            left_index        = extracted_params[i]['left_edge_01_indices'][j]
            right_index       = extracted_params[i]['right_edge_01_indices'][j]
            integral          = integrate_simpson(potential, current, left_index, right_index)
            FW01_integrals[j] = integral
        extracted_params[i]['FWHM_integral'] = FWHM_integrals
        extracted_params[i]['FW01_integral'] = FW01_integrals
    return



def loop_smooth_voltammograms(voltammograms):
    num_grams = len(voltammograms)

    for i in range(num_grams):
        smoothed_current             = smooth_voltammogram(voltammograms[i]['I'].to_numpy())
        voltammograms[i]['I_smooth'] = smoothed_current
    return 


def smooth_voltammogram(voltammogram_current, window_length = 17, polyorder = 6):
    smoothed_current = savgol_filter(voltammogram_current, window_length, polyorder)
    return smoothed_current


def baseline_subtraction(potential, current, algo='snip', expected_FWHM=35):
    baseline_fitter = Baseline(x_data=potential)

    if algo == 'mor':
        spacing = abs(potential[1] - potential[0])
        half_window = expected_FWHM * spacing
        baseline, params = baseline_fitter.mor(current, half_window=half_window)
    else:
        baseline, params = baseline_fitter.snip(current, max_half_window=10, decreasing=True)
    current_baseline_removed = current - baseline
    return current_baseline_removed, baseline

def loop_baseline_subtraction(voltammograms):
    num_grams = len(voltammograms)

    for i in range(num_grams):
        curr_gram = voltammograms[i]
        potential = curr_gram['V'].to_numpy()
        current   = curr_gram['I'].to_numpy()

        # We have to takes negatives for thebaseline algorithms to work, since the signal is "upside down" (i.e. peaks are troughs):
        current_baseline_removed, baseline = baseline_subtraction(potential, -current)

        curr_gram['I_baseline_sub'] = -current_baseline_removed
        curr_gram['Baseline']       = -baseline

    return 



def loop_get_peak_info(voltammograms, lower_bound=-0.35, upper_bound=-0.2, width=10):
    num_grams      = len(voltammograms)
    peak_info_list = []
    for i in range(num_grams):
        curr_gram = voltammograms[i]
        potential = curr_gram['V'].to_numpy()
        current   = curr_gram['I_baseline_sub'].to_numpy()
        peak_info = find_peaks_with_widths(potential, -current, lower_bound, upper_bound, width)
        peak_info_list.append(peak_info)

    return peak_info_list

def find_peaks_with_widths(x, y, lower_bound, upper_bound, width = 20):
    """
    Purpose: Find peaks within specified bounds and return their heights, widths, and indices

    Input:
    - x           (array/float) : x-coordinates of a dataset 
    - y           (array/float) : y-coordinates of a dataset
    - lower_bound (float)       : lower value of x for the "window" in which the function will search for peaks in y 
    - upper_bound (float)       : upper value of x for the "window" in which the function will search for peaks in y
    - prominence  (float)       : The prominence of a peak measures how much a peak stands out from the surrounding 
                                  baseline of the signal and is defined as the absolute vertical distance, in the units
                                  of the data (as opposed to indices) between the peak and its lowest contour line.
    - width       (float)       : minimum acceptable width, in total number of samples, of a peak .... below this value,
                                  peaks will be ignored



    Output:
    - peak_info (dict) 
        - 'peak_indices'          (array/int)   : indices of tagged peaks in y
        - 'peak_heights'          (array/float) : y-values of tagged peaks
        - 'peak_potentials'       (array/float) : x-values of tagged peaks 
        - 'FWHM'                  (array/float) : Full-Width-Half-Max of each of the tagged peaks
        - 'FWHM01'                (array/float) : Full width of each of the tagged peaks at 10% of their height
        - 'left_edge_indices'     (array/int)   : indices of the left edges of the horizontal cross-section of the peaks at half of the apex value
        - 'right_edge_indices'    (array/int)   : indices of the right edges of the horizontal cross-section of the peaks at half of the apex value
        - 'left_edge_01_indices'  (array/int)   : indices of the left edges of the horizontal cross-section of the peaks at 10% of the apex value
        - 'right_edge_01_indices' (array/int)   : indices of the right edges of the horizontal cross-section of the peaks at 10% of the apex value

    """
    mask       = (x >= lower_bound) & (x <= upper_bound) # create a mask where x falls within bounds specified in function call
    x_filtered = x[mask]
    y_filtered = y[mask]

    peaks, properties = find_peaks(y_filtered, width=width, rel_height=0.5, prominence=0.1*(max(y)-min(y)), distance=10) # using find_peaks from scipy.signal ... it returns the index of the peaks


    # Below, we use y instead of y_filtered because we want to consider the entire peak-like structure. The bounds used to obtain
    # y_filtered are merely used so we only pick peaks within a specified region. This region is confined to the neighbourhood
    # of the apex of the peak of interest only, and leaves out much of the peak structure. We still want to consider the full peak
    # structure when we look for the peak widths after tagging the peak.
    corrected_peaks                                          = peaks + np.where(mask)[0][0] # the peak indices were obtained on the filtered data, so we must add back to these indices the number of points prior to
                                                                                            # the filtered range (i.e. the number of points "left" of the cut-off) 
    widths, width_heights, left_edge, right_edge             = peak_widths(y, corrected_peaks, rel_height=0.5)
    widths_01, width_heights_01, left_edge_01, right_edge_01 = peak_widths(y, corrected_peaks, rel_height=0.9) # Note: "rel_height = 0.9" will ensure that we return the peak width at 10% of the peak height

    peak_indices          = corrected_peaks
    left_edge_indices     = np.round(left_edge).astype(int)    
    right_edge_indices    = np.round(right_edge).astype(int)   
    left_edge_01_indices  = np.round(left_edge_01).astype(int)  
    right_edge_01_indices = np.round(right_edge_01).astype(int) 

    peak_info = {
                'peak_indices'          : peak_indices.astype(int),
                'peak_heights'          : y_filtered[peaks], # equivalent to y[peak_indices]
                'peak_potentials'       : x_filtered[peaks], # equivalent to x[peak_indices]
                'FWHM'                  : widths,
                'FW0.1M'                : widths_01,
                'left_edge_indices'     : left_edge_indices,
                'right_edge_indices'    : right_edge_indices,
                'left_edge_01_indices'  : left_edge_01_indices,
                'right_edge_01_indices' : right_edge_01_indices
                }

    return peak_info


def add_peak_info_to_voltammogram_list(voltammogram_list, peak_info_list):
    '''
    Purpose: 

    Input:
    - voltammogram_list (list) : 
    - peak_info_list    (list) :
    '''
    num_grams = len(voltammogram_list)
    for i in range(num_grams):
        voltammogram_list[i]['extracted_peak_parameters'] = peak_info_list[i] # peak_info_list[i] is a dict

    return


def build_analyte_data_table(labels, peak_info, timestamps):
    num_grams  = len(labels)
    # Below: Create a DataFrame from the first peak_info dict within peak_info (a list of dicts).
    # Fortunately, pandas knows how to convert a dict to a DataFrame for us - the keys of the dict 
    # become the colun names of the DataFrame.
    data_table              = pd.DataFrame(peak_info[0]) 
    curr_num_peaks          = len(data_table) # get number of peaks in first peak_info dict (should be 1 but anomalous results happen)
    curr_timestamp          = np.repeat(timestamps[0], curr_num_peaks) # get the timestamp for the first/current peak_info dict and replicate it for each peak
    curr_label              = np.repeat(labels[0], curr_num_peaks) # same idea as curr_timestamp but for the analyte/control label
    data_table['Timestamp'] = curr_timestamp # add a 'Timestamp' column to the data table and populate it with the current/first timestamp corresponding to the peak data
                                             # i.e. with the timestamp corresponding to the peak_info dict that was just converted to a DataFrame
    data_table['Label']     = curr_label

    # Iteratively repeat the above procedure for the remaining analyte peak_info dicts and append to the end of the existing data table
    for i in range(1, num_grams):
        temp_df              = pd.DataFrame(peak_info[i])
        curr_num_peaks       = len(temp_df)
        curr_timestamp       = np.repeat(timestamps[i], curr_num_peaks)
        temp_df['Timestamp'] = curr_timestamp
        curr_label           = np.repeat(labels[i], curr_num_peaks)
        temp_df['Label']     = curr_label
        data_table           = pd.concat([data_table, temp_df], ignore_index=True)


    return data_table





def pickle_voltammogram_info(data_list, save_folder):
    """
    Pickles the given list and saves it in a new folder named 'pickled voltammogram info'
    inside the specified folder path.

    Parameters:
    - data_list: list to be pickled
    - folder_path: path where the 'pickled voltammogram info' folder will be created
    """
    # Define the file path for the pickled file
    file_path = os.path.join(save_folder, "voltammogram_list.pkl")

    # Pickle and save the list
    with open(file_path, "wb") as f:
        pickle.dump(data_list, f)


def pickle_measurement_info(measurement, save_folder):
    """
    Pickles the given measurement dict and saves it in a new folder named 'Full Measurment Info'
    inside the specified folder path.

    Parameters:
    - measurement (dict) : dict to be pickled. Contains all information about a measurement including its voltammogram_list and
                           extracted parameters DataFrame
    - save_folder (str)  : path where the measurement folder will be created
    """
    # Define the file path for the pickled file
    file_path = os.path.join(save_folder, "measurement_info.pkl")

    # Pickle and save the list
    with open(file_path, "wb") as f:
        pickle.dump(measurement, f)




def loop_cut_leading_points(voltammograms, num_to_cut=0):
    num_grams = len(voltammograms)
    for i in range(num_grams):
        cut_voltammogram_leading_points(voltammograms[i], num_to_cut)        

    return 

def cut_voltammogram_leading_points(df_voltammogram, num_cut = 0):
    '''
    Purpose: To remove the leading points at the beginning of a DataFrame. 
             In the context of a voltammogram there is often a large background 
             charging current in this region that needs to be removed before 
             baseline subtraction and smoothing.

    Input:
    - df_voltammogram (DataFrame) : A DataFrame object containing the potential and current information of a voltammogram.
    - num_pts   (int)             : The number of row to remove from the beginning of the DataFrame - i.e. the beginning/charging 
                                    region of the voltammogram

    Effect:
    - df_voltammogram is altered to drop the first num_cut rows, which is done with the intention of removing the charging 
      current region of a scan.
    '''

    num_pts = len(df_voltammogram)
    # Check if num_cut is valid. It must be less than the data length and greater than zero:
    if num_cut < num_pts and num_cut > 0:
        df_voltammogram.drop(index=df_voltammogram.index[:num_cut], inplace=True)  # Removes first num_cut rows
        df_voltammogram.reset_index(drop=True, inplace=True)          # Resets index after dropping
    else:
        return # Reject alteration of the DataFrame if num_cut is invalid.
    
    return


import os

def save_processed_voltammograms(save_folder, voltammograms, voltammogram_files):
    """
    Purpose: To save a list of processed voltammograms (DataFrames) to csv files within a specified folder.
             The voltammograms are processed versions of voltammograms that were loaded from voltammogram_files.
             A suffix - "PROCESSED" - is appended to the altered voltammograms when saved, such that they are 
             simulatenously differentiable and identifiable with their original unaltered files.

    Input:
    - save_folder (str)         : The path to the parent folder where the voltammograms will be saved as csv files
    - voltammograms (list)      : a list of DataFrames which contain processed voltammograms
    - voltammogram_files (list) : a list of the filenames (str) corresponding to the original unprocessed versions
                                  of "voltammograms"
    
    """
    num_grams = len(voltammograms)
    for i in range(num_grams):
        voltammograms[i].to_csv(save_folder + "\\" + voltammogram_files[i][:-4] + "_PROCESSED" + ".csv")
    
    return 


def make_subfolder(parent_folder, subfolder):
    """
    Purpose:
    Creates a subdirectory with name specified by the value of subfolder, within the specified parent folder.
    If the directory already exists, it appends a numeric suffix (e.g., '_1', '_2') to make the directory name unique.
    
    Input:
    - parent_folder (str) : The path to the parent folder where the subdirectory will be created.
    - subfolder     (str) : the desired name of the new subdirectory/subfolder
    
    Output:
    new_dir_path (str): The path to the created subdirectory, which is unique in the parent folder.
    """
    dir_name = subfolder
    counter  = 1
    
    # Check if the directory already exists, if so, append a number to the directory name
    while os.path.exists(os.path.join(parent_folder, dir_name)):
        dir_name = f"{subfolder}_{counter}"
        counter += 1
    
    # Create the directory
    new_dir_path = os.path.join(parent_folder, dir_name)
    os.makedirs(new_dir_path)

    return new_dir_path

def get_average_baseline(potential, baseline):
    num_samples = len(baseline)
    low_index   = 0
    upp_index   = num_samples-1
    avg         = integrate_simpson(potential, baseline, low_index, upp_index) / num_samples


    return avg


def loop_get_average_baseline(voltammogram_list):
    num_grams = len(voltammogram_list)

    all_baseline  = [x['voltammogram_processed']['Baseline'].to_numpy() for x in voltammogram_list]
    all_potential = [x['voltammogram_processed']['V'].to_numpy() for x in voltammogram_list]
    for i in range(num_grams):
        baseline     = all_baseline[i]
        potential    = all_potential[i]
        avg_baseline = get_average_baseline(potential, baseline)
        
        voltammogram_list[i]['average_baseline'] = avg_baseline

    return

def loop_integrate_peak_baselines(extracted_peak_params, voltammograms_processed):
    num_grams = len(voltammograms_processed)

    for i in range(num_grams):
        num_peaks = len(extracted_peak_params[i]['peak_indices'])
        potential = voltammograms_processed[i]['V'].to_numpy()
        current   = voltammograms_processed[i]['Baseline'].to_numpy()

        FWHM_integrals = np.zeros(num_peaks)
        FW01_integrals = np.zeros(num_peaks)
        for j in range(num_peaks):

            # Integrate peak over range spanning > 0.5 Max
            left_index        = extracted_peak_params[i]['left_edge_indices'][j]
            right_index       = extracted_peak_params[i]['right_edge_indices'][j]
            integral          = integrate_simpson(potential, current, left_index, right_index)
            FWHM_integrals[j] = integral


            # Integrate peak over range spanning > 0.1 Max
            left_index        = extracted_peak_params[i]['left_edge_01_indices'][j]
            right_index       = extracted_peak_params[i]['right_edge_01_indices'][j]
            integral          = integrate_simpson(potential, current, left_index, right_index)
            FW01_integrals[j] = integral
        extracted_peak_params[i]['baseline_FWHM_integral'] = FWHM_integrals
        extracted_peak_params[i]['baseline_FW01_integral'] = FW01_integrals

    return


import pickle
import os

def save_sweep_info_to_files(dictionaries_list, filenames, directory):
    """
    Purpose: Saves a list of dictionaries to files using pickle. We use the original filenames from which the
             raw scan data (i.e. voltammograms) were loaded, and then add the suffix '_Sweep'
    
    Input:
    - dictionaries_list (list): List of dictionaries to save.
    - filenames         (list): List of filenames (strings) corresponding to each dictionary.
    - directory         (str) : Directory where the files should be saved.
    
    Raises:
    - ValueError       : If the lengths of dicts and filenames don't match.
    - FileNotFoundError: If the directory does not exist.
    """
    if len(dictionaries_list) != len(filenames):
        raise ValueError("The number of dictionaries must match the number of filenames.")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    for data, filename in zip(dictionaries_list, filenames):
        filepath = os.path.join(directory, filename[:-4] + '_Sweep.pkl') # drop .csv from filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    return 

import pandas as pd

def add_scan_order(df, rank_column):
    """
    Adds a 'Scan Order' column to the dataframe based on the ranking of values in rank_column.
    Equal values receive the same rank, but the rank increment is by 1 instead of the number of ties.

    Parameters:
    - df (pd.DataFrame): The dataframe to modify.
    - rank_column (str): The column whose values will be ranked.

    Returns:
    - df (pd.DataFrame): The modified dataframe with the new 'Scan Order' column.
    """
    if rank_column not in df.columns:
        raise ValueError(f"Column '{rank_column}' not found in dataframe.")
    
    # Sort values by the rank column to ensure we can handle ties
    sorted_df  = df.sort_values(by='Timestamp', ascending=True)
    df_thc     = sorted_df[sorted_df['Label'] == 'THC'].reset_index(drop=True)
    df_control = sorted_df[sorted_df['Label'] == 'Control'].reset_index(drop=True)

    def make_rank_column(sorted_df, name, rank_column):
        # Initialize the rank
        current_rank = 1
        last_value   = None
        new_column   = []
        for i, row in sorted_df.iterrows():
            # If the current value is equal to the last value, keep the same rank
            if row[rank_column] == last_value:
                new_column.append(current_rank-1)
            else:
                new_column.append(current_rank)
                current_rank += 1  # Increment rank for the next unique value
            last_value = row[rank_column]

        sorted_df[name] = new_column
    
    make_rank_column(df_thc, 'Sweep', 'Timestamp')
    make_rank_column(df_control, 'Sweep', 'Timestamp')

    df = pd.concat([df_thc, df_control])

    return df



def create_ranked_pairplot(df, rank_column, color_palette='Blues'):
    """
    Creates a pairplot with numeric columns, using one column as rank (hue).
    Returns the plot object for further customization or saving.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    rank_column : str
        Numeric column to use as hue
    color_palette : str
        Sequential color palette (default: 'Blues')
        
    Returns:
    --------
    seaborn.axisgrid.PairGrid
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    plot_columns = [col for col in numeric_cols if col != rank_column]
    
    if rank_column not in numeric_cols:
        raise ValueError(f"Rank column '{rank_column}' must be numeric")
    
    g = sns.pairplot(
        df,
        vars=plot_columns,
        hue=rank_column,
        palette=color_palette,
        plot_kws={'alpha': 0.7, 'edgecolor': 'none'},
        diag_kws={'alpha': 0.7, 'edgecolor': 'none'}
    )
    
    g.fig.suptitle(f'Pairplot with {rank_column} as hue', y=1.02)
    g.add_legend(title=rank_column)
    
    return g

def create_hue_pairplot(df, hue_column, selected_columns=None, alpha=0.9, kde_alpha=0.5):
    """
    Create a pairplot with Pearson correlation coefficients for each hue category.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data to plot
    hue_column : str
        Column name to use for hue grouping
    selected_columns : list (default=None)
        List of columns to include in the pairplot. If None, uses all numeric columns.
    alpha : float (default=0.9)
        Transparency level for the data points
    kde_alpha : float (default=0.5)
        Transparency level for the KDE contours
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the pairplot
    """
    # Get unique categories for hue
    hue_categories = df[hue_column].unique()
    num_categories = len(hue_categories)
    palette = sns.color_palette(n_colors=num_categories)
    
    # Get numeric columns (excluding hue column)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if hue_column in numeric_columns:
        numeric_columns.remove(hue_column)
    
    # Filter to selected columns if provided
    if selected_columns is not None:
        # Ensure selected_columns exists in dataframe and are numeric
        valid_columns = [col for col in selected_columns 
                        if col in numeric_columns and col in df.columns]
        if not valid_columns:
            raise ValueError("No valid numeric columns selected")
        numeric_columns = valid_columns
    
    # Create pairplot with only selected columns
    plot_data = df[numeric_columns + [hue_column]]
    g = sns.pairplot(plot_data, hue=hue_column, palette=palette,
                    plot_kws={'alpha': alpha},
                    diag_kind='kde')
    
    # Add KDE contours to lower triangular plots
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            if i > j:  # Lower triangular
                ax = g.axes[i, j]
                
                for k, category in enumerate(hue_categories):
                    subset = df[df[hue_column] == category]
                    x_col = numeric_columns[j]
                    y_col = numeric_columns[i]
                    
                    x = subset[x_col].dropna()
                    y = subset[y_col].dropna()
                    
                    if len(x) >= 2 and len(y) >= 2:
                        sns.kdeplot(x=x, y=y, ax=ax,
                                  color=palette[k],
                                  alpha=kde_alpha,
                                  levels=5,
                                  linewidths=1)
    
    # Add correlations to upper triangular plots
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            if i < j:  # Upper triangular
                ax = g.axes[i, j]
                
                for k, category in enumerate(hue_categories):
                    subset = df[df[hue_column] == category]
                    x_col = numeric_columns[i]
                    y_col = numeric_columns[j]
                    
                    x = subset[x_col]
                    y = subset[y_col]
                    valid_indices = ~(x.isna() | y.isna())
                    x_clean = x[valid_indices]
                    y_clean = y[valid_indices]
                    
                    if len(x_clean) >= 2:
                        corr, _ = pearsonr(x_clean, y_clean)
                        ax.text(0.05, 0.9 - (k * 0.1),
                               f"{category}: r={corr:.2f}",
                               transform=ax.transAxes,
                               color=palette[k],
                               fontsize=8,
                               fontweight='bold')
    
    return g.fig




def select_columns(df):
    """
    Creates a tkinter window to select numeric columns and a categorical hue column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to select columns from
        
    Returns:
    --------
    tuple: (list of selected columns, selected hue column)
    """
    # Create main window
    root = tk.Tk()
    root.title("Column Selection")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Variables to store selections
    selected_columns = []
    hue_var = tk.StringVar()
    checkbutton_vars = {}  # Store IntVars for checkboxes
    
    # Frame for column selection
    selection_frame = ttk.LabelFrame(root, text="Select Numeric Columns", padding=10)
    selection_frame.pack(padx=10, pady=5, fill='both', expand=True)
    
    # Create checkbuttons for each numeric column
    for col in numeric_cols:
        var = tk.IntVar(value=0)  # Initialize unchecked (0)
        checkbutton_vars[col] = var  # Store the variable
        
        cb = tk.Checkbutton(
            selection_frame,
            text=col,
            variable=var,  # Link to the IntVar
            onvalue=1,
            offvalue=0
        )
        cb.pack(anchor='w')
    
    # Button frame for "Select All" and "Deselect All"
    button_frame = tk.Frame(selection_frame)
    button_frame.pack(pady=5)
    
    # Select All button (checks all checkboxes)
    def select_all():
        for var in checkbutton_vars.values():
            var.set(1)  # Set to checked (1)
    
    select_all_btn = ttk.Button(
        button_frame,
        text="Select All",
        command=select_all
    )
    select_all_btn.pack(side='left', padx=5)
    
    # Deselect All button (unchecks all checkboxes)
    def deselect_all():
        for var in checkbutton_vars.values():
            var.set(0)  # Set to unchecked (0)
    
    deselect_all_btn = ttk.Button(
        button_frame,
        text="Deselect All",
        command=deselect_all
    )
    deselect_all_btn.pack(side='left', padx=5)
    
    # Hue selection dropdown
    hue_frame = ttk.LabelFrame(root, text="Select Hue Column", padding=10)
    hue_frame.pack(padx=10, pady=5, fill='x')
    
    if categorical_cols:
        hue_dropdown = ttk.Combobox(
            hue_frame,
            textvariable=hue_var,
            values=categorical_cols,
            state="readonly"
        )
        hue_dropdown.pack(pady=5)
        hue_dropdown.current(0)  # Select first item by default
    else:
        ttk.Label(hue_frame, text="No categorical columns found").pack()
    
    # Confirm button
    def on_confirm():
        nonlocal selected_columns
        selected_columns = [col for col, var in checkbutton_vars.items() if var.get() == 1]
        root.destroy()
    
    confirm_btn = ttk.Button(
        root,
        text="Confirm",
        command=on_confirm
    )
    confirm_btn.pack(pady=10)
    
    # Run the window
    root.mainloop()
    
    return selected_columns, hue_var.get() if categorical_cols else None


def create_numeric_pairplot(df, hue_column, columns=None, palette='viridis'):
    """
    Creates a pairplot using only numeric columns, with hue that can be ANY dtype.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    hue_column : str
        Column to use for hue coloring (can be numeric OR categorical)
    columns : list, optional
        Specific numeric columns to include (default: all numeric)
    palette : str or list
        Color palette to use
    """
    # Select only numeric columns for plotting (excluding hue_column from auto-selection)
    numeric_cols = df.select_dtypes(include=['number']).columns
    plot_columns = columns if columns else [col for col in numeric_cols if col != hue_column]
    
    # Create PairGrid (hue_column can be any dtype)
    g = sns.PairGrid(df, vars=plot_columns, hue=hue_column, palette=palette)
    
    # Lower triangle: Scatter + KDE
    g.map_lower(sns.scatterplot, alpha=0.7, edgecolor='none')
    g.map_lower(sns.kdeplot, levels=5, alpha=0.3, linewidths=1)
    
    # Upper triangle: Scatter + Pearson R
    g.map_upper(sns.scatterplot, alpha=0.7, edgecolor='none')
    g.map_upper(_add_correlation)  # Only shown for numeric-numeric pairs
    
    # Diagonal: Histogram/KDE for numeric, countplot for categorical hue
    if pd.api.types.is_numeric_dtype(df[hue_column]):
        g.map_diag(sns.histplot, kde=True, alpha=0.7, edgecolor='none')
    else:
        g.map_diag(sns.histplot, alpha=0.7, edgecolor='none', kde=False)
    
    g.fig.suptitle(f'Pairplot with {hue_column} as hue', y=1.02)
    g.add_legend(title=hue_column)
    
    return g

def _add_correlation(x, y, **kwargs):
    """Helper to add Pearson R to upper triangle"""
    ax = plt.gca()
    r, _ = stats.pearsonr(x, y)
    ax.annotate(f"r = {r:.2f}", 
                xy=(0.5, 0.9), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_peak_coords_plot(extracted_params_table, xlabel='x', ylabel='y', title='y v.s. x'):
    x_data = extracted_params_table['peak_potentials'].to_numpy()
    y_data = extracted_params_table['peak_heights'].to_numpy()
    z_data = extracted_params_table['Label'].to_numpy()

    fig, ax = create_kde_scatterplot(x_data, y_data, z_data)
    ek.make_pretty(ax, xTitle=xlabel, yTitle=ylabel, title=title)
    return fig

def create_kde_scatterplot(x, y, z, colors=None,
                         alpha=0.7, figsize=(16.10, 10), point_size=150):
    """
    Creates a scatterplot with KDE contours that can be saved with save_plot().
    
    Parameters:
    -----------
    x, y : array-like
        Data positions
    z : array-like
        Categorical group labels
    colors : list
        Colors for each group
    levels : list
        Density levels for contours
    alpha : float
        Point transparency
    figsize : tuple
        Figure size
    point_size : float
        Size of scatter points
        
    Returns:
    --------
    fig : matplotlib Figure object
        Ready for saving with save_plot()
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Get unique groups
    groups = np.unique(z)
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    
    # Set default colors
    if colors is None:
        colors = sns.color_palette('bright', n_colors=len(groups))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each group
    for i, group in enumerate(groups):
        mask = z == group
        x_vals = x[mask]
        y_vals = y[mask]
        
        # Scatter plot
        ax.scatter(x_vals, y_vals, color=colors[i],
                  label=group, alpha=alpha, s=point_size)
        
        # KDE contours
        if len(x_vals) > 1:  # Need at least 2 points
            kde = gaussian_kde(np.vstack([x_vals, y_vals]))
            
            # Create grid with buffer
            x_buf = 0.2 * (x_vals.max() - x_vals.min())
            y_buf = 0.2 * (y_vals.max() - y_vals.min())
            xx, yy = np.mgrid[x_vals.min()-x_buf:x_vals.max()+x_buf:100j,
                              y_vals.min()-y_buf:y_vals.max()+y_buf:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)
            
            # Contours with level labels
            contours = ax.contour(xx, yy, density, levels=2,
                                 colors=[colors[i]], linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=8, 
                     fmt=lambda x: f"{x:.3f}")
    
    # Add labels and legend
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(title="Groups")
    plt.tight_layout()
    
    return fig, ax

def save_plot(fig, save_path, filename='plot.png', dpi=300):
    """
    Saves a matplotlib figure as PNG.
    
    Parameters:
    -----------
    fig : matplotlib Figure
        Figure to save
    save_path : str
        Directory path
    filename : str
        Output filename
    dpi : int
        Image resolution
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")



def labeled_colored_scatterplot(
    x, y, z, label,
    colormaps=None,
    min_color_frac=0.2,
    max_color_frac=1.0,
    title='',
    figsize=(16.18, 10),
    marker_size=150,
    edgecolor='black',
    alpha=1.0,
    grid=False,
    xlabel='x',
    ylabel='y',
    zlabel=None

):
    """
    Creates a scatter plot with multiple color gradients for different label categories.
    
    Parameters:
    x, y, z: Data arrays
    label: Array of category labels
    colormaps: List of colormap names (one per unique label). If None, uses default sequence.
    min_color_frac: Start of colormap range (0-1)
    max_color_frac: End of colormap range (0-1)
    xlabel, ylabel: Axis labels
    title: Plot title
    figsize: Figure dimensions
    marker_size: Size of scatter points
    edgecolor: Color of point edges
    alpha: Transparency
    grid: Whether to show grid
    
    Returns:
    matplotlib.figure.Figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels and prepare colormaps
    unique_labels = np.unique(label)
    if colormaps is None:
        # Default colormap sequence
        colormaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys']
    
    # Plot each category
    for i, lbl in enumerate(unique_labels):
        # Get subset of data for this label
        mask = label == lbl
        x_sub = x[mask]
        y_sub = y[mask]
        z_sub = z[mask]
        
        # Get colormap (cycle if needed)
        cmap_name = colormaps[i % len(colormaps)]
        cmap = plt.get_cmap(cmap_name)
        truncated_cmap = LinearSegmentedColormap.from_list(
            f'truncated_{cmap_name}',
            cmap(np.linspace(min_color_frac, max_color_frac, 256)))
        
        # Plot this category
        sc = ax.scatter(
            x_sub, y_sub, c=z_sub,
            cmap=truncated_cmap,
            edgecolor=edgecolor,
            s=marker_size,
            alpha=alpha,
            label=str(lbl)  # For potential legend
        )
        
        # Add colorbar for this category
        cbar = fig.colorbar(sc, ax=ax)
        if zlabel:
            cbar.set_label(str(lbl) + '-' + zlabel)
        else:
            cbar.set_label(str(lbl))
    
    # Style the plot:
    ek.make_pretty(ax, 
                    xTitle=xlabel, 
                    yTitle=ylabel, 
                    title=title)
    ax.set_facecolor('whitesmoke')
    return fig, ax


def plot_current_potential_vs_time(extracted_params_table):

    potential = extracted_params_table['peak_potentials'].to_numpy()
    current   = extracted_params_table['peak_heights'].to_numpy()
    time      = extracted_params_table['Timestamp'].to_numpy()
    label     = extracted_params_table['Label'].to_numpy()

    y_colour=['navy', 'darkgreen', 'purple', 'lime']
    x_colour=['crimson', 'orange', 'pink', 'dodgerblue']
    y_marker=['^', 's', 'o']
    x_marker=['D', 'P', 'X']

    fig, axes = dual_axis_plot(
                                time, current, potential, label,
                                y_color=y_colour,
                                x_color=x_colour,
                                y_marker=y_marker,
                                x_marker=x_marker,
                                y_label='Current\n(nA)', 
                                x_label= 'Potential\n(V)',
                                t_label='Time\n(s)', 
                                title='Individual Time-Drift of Current and Potential'
                                )

    # Improving cosmetics:
    for ax in axes:
        ek.make_pretty(ax[0])

    return fig, axes

def dual_axis_plot(t, y, x, z=None, y_color='blue', x_color='red', 
                   y_marker='o', x_marker='s', 
                   y_label='y', x_label='x', t_label='t',
                   title='Dual Axis Plot'):
    """
    Creates stacked dual-axis plots grouped by z values.
    
    Parameters:
        t: Array for horizontal axis
        y: Array for left vertical axis
        x: Array for right vertical axis
        z: Array indicating group membership (None for no grouping)
        y_color, x_color: Colors for y and x data (can be single color or list by group)
        y_marker, x_marker: Marker styles (single or list by group)
        y_label, x_label, t_label: Axis labels
        title: Plot title
        
    Returns:
        fig: matplotlib.figure.Figure object
        axes: List of axes tuples (ax1, ax2) for each subplot
    """
    # Handle grouping
    if z is None:
        groups = [None]
        group_labels = ['']
        n_groups = 1
    else:
        unique_z = np.unique(z)
        groups = unique_z
        group_labels = [f'Group {g}' for g in groups]
        n_groups = len(groups)
    
    # Create figure with stacked subplots
    fig, all_axes = plt.subplots(n_groups, 1, figsize=(16.18, 10*n_groups), sharex=True)
    if n_groups == 1:
        all_axes = [all_axes]  # Ensure it's always a list
    
    # Handle color/marker options (convert single values to lists)
    if isinstance(y_color, str):
        y_colors = [y_color] * n_groups
    else:
        y_colors = y_color
        
    if isinstance(x_color, str):
        x_colors = [x_color] * n_groups
    else:
        x_colors = x_color
        
    if isinstance(y_marker, str):
        y_markers = [y_marker] * n_groups
    else:
        y_markers = y_marker
        
    if isinstance(x_marker, str):
        x_markers = [x_marker] * n_groups
    else:
        x_markers = x_marker
    
    axes_list = []
    for i, (group, ax) in enumerate(zip(groups, all_axes)):
        # Select data for this group
        if group is not None:
            mask = z == group
            t_group = t[mask]
            y_group = y[mask]
            x_group = x[mask]
        else:
            t_group = t
            y_group = y
            x_group = x
        
        # Left axis (y data)
        ax1 = ax
        ax1.set_ylabel(y_label, color=y_colors[i])
        y_plot = ax1.scatter(t_group, y_group, 
                            color=y_colors[i], 
                            marker=y_markers[i], 
                            label=y_label,
                            s=150
                            )
        ax1.tick_params(axis='y', labelcolor=y_colors[i])
        
        # Right axis (x data)
        ax2 = ax1.twinx()
        ax2.set_ylabel(x_label, color=x_colors[i])
        x_plot = ax2.scatter(t_group, x_group, 
                            color=x_colors[i], 
                            marker=x_markers[i], 
                            label=x_label,
                            s=150
                            )
        ax2.tick_params(axis='y', labelcolor=x_colors[i])
        
        # Add group label if needed
        if group is not None:
            ax1.text(0.02, 0.9, group_labels[i], 
                    transform=ax1.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Combine legends from both axes
        lines = [y_plot, x_plot]
        ax1.legend(lines, [line.get_label() for line in lines], 
                 bbox_to_anchor=(1, 1), loc='upper left')
        
        axes_list.append((ax1, ax2))
    
    # Final figure adjustments
    all_axes[-1].set_xlabel(t_label)
    fig.suptitle(title)
    fig.tight_layout()
    
    return fig, axes_list

def plot_with_tagged_points(ax, x, y, peak_indices, **kwargs):
    """Plot y vs x and mark peaks with 'x' markers (no legend entries)."""
    ax.scatter(x, y, s=120, **kwargs)
    ax.plot(x, y, **kwargs)
    for idx in peak_indices:
        if 0 <= idx < len(x):
            ax.scatter(x[idx], y[idx], marker='x', c='red', s=500, linewidths=2,
                       label='_nolegend_')  # Hide all peak labels
    return ax


def overlay_scans(voltammogram_list):
    '''
    Purpose: This function is designed to overlay all scans common to a 
            single measurement. The peaks for each scan are tagged. The 
            order in which the scans were run is visualized as a colour
            factor over the data, such that earlier scans are lighter 
            coloured while later scans are darker. The intention is to 
            visualize time-drift in the voltammograms.

    Input:
    - voltammogram_list (list) : a list of dictionaries, each of which contains all info relevant to a given scan

                              
    Output:
    - fig (matplotlib figure) : a figure of overlayed voltammograms with tagged peaks
    - ax (matplotlib axes)    : axesobject corresponding to fig
    '''
    unique_labels = np.unique([x['qualifiers']['Label'] for x in voltammogram_list])

    figures = []
    axes    = []
    for label in unique_labels:
        mask = [x['qualifiers']['Label'] == label for x in voltammogram_list]
        indices = np.where(mask)[0]
        voltammogram_list_filtered = [voltammogram_list[i] for i in indices]


        x_data       = [x['voltammogram_processed']['V'].to_numpy() for x in voltammogram_list_filtered]
        y_data       = [x['voltammogram_processed']['I_baseline_sub'].to_numpy() for x in voltammogram_list_filtered]
        peaks_list   = [x['extracted_peak_parameters']['peak_indices'] for x in voltammogram_list_filtered]
        time_values  = [x['qualifiers']['Timestamp'] for x in voltammogram_list_filtered]

        # Set up the figure
        fig, ax = plt.subplots(figsize=(16.18,10))

        # Define colormap range
        cmap_range = (0.1, 1.0)  # Restricting range to remove white 
        cmap       = plt.get_cmap('Greens')

        # Create normalization for the data
        norm_data = Normalize(vmin=min(time_values), vmax=max(time_values))

        # Loop through datasets
        for x, y, peaks, time in zip(x_data, y_data, peaks_list, time_values):
            # Normalize time to [0,1], then scale to cmap_range
            normalized_time = norm_data(time)
            scaled_time = cmap_range[0] + (cmap_range[1] - cmap_range[0]) * normalized_time
            color = cmap(scaled_time)
            plot_with_tagged_points(ax, x, y, peaks, color=color, label='_nolegend_')

        # Create colorbar with actual time values but restricted color range
        # We need to create a custom colormap that only shows our desired range
        subset_cmap = ListedColormap(cmap(np.linspace(cmap_range[0], cmap_range[1], 256)))
        sm = ScalarMappable(norm=norm_data, cmap=subset_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Time (arbitrary units)')

        # Add 'Peaks' to legend
        ax.scatter([], [], marker='x', c='red', s=500, linewidths=2, label='{} Peaks'.format(label))
        ax.legend(loc='upper left', bbox_to_anchor=(0.05,-0.1), fontsize=16)

        # Cosmetics:
        ek.make_pretty(ax, xTitle='Potential\n(V)',
                        yTitle='Current\n(nA)', 
                        title='Overlayed Scans with Time as Colour-Factor\n(Label : {})'.format(label))
        figures.append(fig)
        axes.append(ax)


    return figures, axes, unique_labels


def get_analysis_mode():
    """
    Opens a tkinter window to let the user choose between 'batch' and 'single measurement' modes.
    Returns either 'batch' or 'single' based on user selection.
    """
    choice = None
    
    def on_submit():
        nonlocal choice
        choice = mode_var.get()
        root.destroy()
    
    # Create main window
    root = tk.Tk()
    root.title("Measurement Mode Selection")
    root.resizable(False, False)
    
    # Use themed widgets for a modern look
    style = ttk.Style()
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 11))
    style.configure('TRadiobutton', background='#f0f0f0', font=('Helvetica', 10))
    
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid()
    
    # Header
    ttk.Label(main_frame, text="Select Measurement Mode", 
              font=('Helvetica', 12, 'bold')).grid(row=0, column=0, pady=(0, 15), columnspan=2)
    
    # Mode selection using radio buttons with icons
    mode_var = tk.StringVar(value='single')  # default selection
    
    # Single measurement option
    single_frame = ttk.Frame(main_frame)
    single_frame.grid(row=1, column=0, padx=10, pady=5, sticky='w')
    ttk.Radiobutton(
        single_frame, 
        variable=mode_var, 
        value='single',
        text="Single"
    ).grid(row=0, column=0, sticky='w')
    ttk.Label(
        single_frame, 
        text="Process a single measurement.", 
        foreground='#555555'
    ).grid(row=1, column=0, sticky='w', padx=20)
    
    # Batch option
    batch_frame = ttk.Frame(main_frame)
    batch_frame.grid(row=2, column=0, padx=10, pady=5, sticky='w')
    ttk.Radiobutton(
        batch_frame, 
        variable=mode_var, 
        value='batch',
        text="Batch Processing"
    ).grid(row=0, column=0, sticky='w')
    ttk.Label(
        batch_frame, 
        text="Process multiple measurements sequentially", 
        foreground='#555555'
    ).grid(row=1, column=0, sticky='w', padx=20)
    
    # Submit button
    submit_btn = ttk.Button(
        main_frame, 
        text="Confirm Selection", 
        command=on_submit
    )
    submit_btn.grid(row=3, column=0, pady=(15, 5), sticky='ew')
    
    # Center the window and run
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

    return choice




import tkinter as tk

import tkinter as tk

def choose_data_vis_options():
    '''
    Purpose: This function creates a Tkinter window displaying a vertically stacked list of options, 
             each with its own selectable radio button (used like a checkbox). The user can choose 
             any combination of the available options, use a "Select All" button to check them all, 
             and confirm their choices by clicking a "Confirm Choices" button. The function returns 
             a dictionary of booleans indicating which options were selected at confirmation.

    Input:
    - None

    Output:
    - selections_dict (dict) : a dictionary where keys are option names (str) and values are booleans 
                               (True if selected, False otherwise)
    '''
    options = [
        "pairplot",
        "2D_peak_coords_drift",
        "1D_peak_coords_drift",
        "overlay_tagged_sweeps"
    ]

    selections_dict = {}

    def on_confirm():
        nonlocal selections_dict
        selections_dict = {opt: var.get() for opt, var in zip(options, bool_vars)}
        root.destroy()

    def select_all():
        for var in bool_vars:
            var.set(True)

    root = tk.Tk()
    root.title("Select Plot Options")

    bool_vars = []
    for option in options:
        var = tk.BooleanVar(value=False)
        bool_vars.append(var)
        frame = tk.Frame(root)
        frame.pack(anchor="w", padx=10, pady=2)
        radio = tk.Radiobutton(frame, variable=var, value=True)
        radio.pack(side="left")
        label = tk.Label(frame, text=option)
        label.pack(side="left")

    select_all_btn = tk.Button(root, text="Select All", command=select_all)
    select_all_btn.pack(pady=5)

    confirm_btn = tk.Button(root, text="Confirm Choices", command=on_confirm)
    confirm_btn.pack(pady=5)

    root.mainloop()
    return selections_dict


def add_measurement_tags(all_measurements, all_extracted_params_tables):
    '''
    Purpose: The function adds a tag to the extracted parameter tables such that, when they are concatenated
             in a batch analysis, the different measurements, with otherwise similar data (e.g. timestamp,
             date, etc may all be the same), can be differentiated. We're adding information which will make
             all data distinguishable.

    Input:
    - all_measurements (list)            : list of lists of dictionaries storing all information about voltammograms/scans for each measurement
    - all_extracted_params_tables (list) : A list of pandas DataFrames which contain information about 
                                           the voltammograms stored in all_measurements.  
    '''
    num_meas = len(all_measurements)

    for i in range(num_meas):
        voltammogram_list = all_measurements[i]
        all_extracted_params_tables[i]['Measurement'] = np.repeat(voltammogram_list[0]['qualifiers']['Q1'] ,len(all_extracted_params_tables[i]))            