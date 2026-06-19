
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Section: Import libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
import Ekidna as ek

import pipeline_functions as pipeline
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Section: Load data
analysis_mode = pipeline.get_analysis_mode() # choose single measurement or batch analysis

# Get path to folder storing voltammograms saved as csv files. It should be a folder of voltammograms of identical 
# formats (i.e. same columns and same column lengths between different csv files).
parent_folderpath = pipeline.select_folder()

if analysis_mode == 'single':
    all_measurements = [pipeline.load_single_measurement(parent_folderpath)] # put into a list for compatibility with batch approach

else:
    all_measurements, subfolders = pipeline.load_many_measurements(parent_folderpath) 

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Section: Choose which data visualization algorithms to apply. The method returns a dict. Elements are boolean
# and, when True, indicate later on in the program to run the corresponding data visualization procedure. When
# False, that particular procedure is skipped. There is one boolean for each data visualization procedure.
data_vis_choices = pipeline.choose_data_vis_options() 


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Section: Loop over measurements, applying standard analysis pipeline to each:
num_measurements            = len(all_measurements)
all_extracted_params_tables = []
for i in range(num_measurements):

    # To keep user aprised of progress:
    print('########################################################')
    print('########################################################')
    print('########################################################')
    print('ANALYZING MEASUREMENT {} OF {} . . . '.format(i+1, num_measurements))
    print('\n')


    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Load all measurements
    if analysis_mode == 'batch':
        folderpath = parent_folderpath + '\\' + subfolders[i]
    else:
        folderpath = parent_folderpath


    voltammogram_list = all_measurements[i] 


    voltammograms_raw       = [x['voltammogram'] for x in voltammogram_list]  # Put voltammograms into a list
    voltammogram_files      = [x['file'] for x in voltammogram_list]          # Put filenames into a list
    num_grams               = len(voltammograms_raw)                          # Get the total number of voltammograms
    voltammograms_processed = voltammograms_raw.copy()                        # We make a copy of the list of voltammograms.
                                                                              # These copies will be processed and altered in the analysis

    pipeline.add_voltammogram_copies_to_list(voltammogram_list, voltammograms_processed)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Parse filenames to extract qualifiers/parameters/info
    # We create a list of lists. The sublists contain various qualifying parameters like the date, timestamps of the 
    # subscans within a measurement, analyte, etc . . . :
    qualifiers_list   = pipeline.loop_parse_filenames(voltammogram_files) 

    pipeline.add_qualifiers_to_voltammogram_list(voltammogram_list, qualifiers_list)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Choose number of points to cut from beginning of each scan in a measurement.
    # We only look at scans from the first measurement and assume the rests of the
    # measurements will require a similar cut-off (we assume the first measurement
    # is representative of the general form of the rest of the measurements):
    if i == 0:
        num_to_cut = pipeline.choose_cutoff_with_visual_assist(voltammograms_processed)
    pipeline.loop_cut_leading_points(voltammograms_processed, num_to_cut)



    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Smooth data using the savgol filter
    pipeline.loop_smooth_voltammograms(voltammograms_processed)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Apply baseline subtraction to smoothed data
    pipeline.loop_baseline_subtraction(voltammograms_processed)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Create a subfolder in which to save results of this analysis.
    ## At this point in the script, we start saving results in various forms including csv files of processed
    ## voltammograms, data tables, pickled objects, etc. We create subfolder, within the folder from which we've loaded
    ## the data being analyzed, in which to store our results.
    analysis_directory = pipeline.make_subfolder(folderpath, 'Analysis') # creates a directory called "Analysis" in the folderpath directory

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # Section: Save processed voltammograms to csv
    ## By this point in the script, we've added the smoothed current, and the smoothed-baseline-subtracted current to the
    ## voltammogram dataframes. We export these adjusted frames, as csv files, to the folder from which the original 
    ## voltammograms were imported. We use the same filenames but add the suffix "PROCESSED" to distinguish from the 
    ## original unprocessed data.
    processed_path = pipeline.make_subfolder(analysis_directory, 'Processed Voltammograms')
    processed_path = pipeline.save_processed_voltammograms(processed_path, voltammograms_processed, voltammogram_files)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    # SECTION: Extract parameters from smoothed-baseline-subtracted voltammograms
    ## Parameters are extracted via peak-finding, integration of peaks, baseline averaging, and other techniques.

    # Get peak coordinates, indices, widths and add them as an entry called 'peak_info' in the sub-dicts within voltammogram_list.
    # That is, each dict that the info for a given voltammogram will now contain a new entry with extracted peak information.
    peak_info_list = pipeline.loop_get_peak_info(voltammograms_processed)
    pipeline.add_peak_info_to_voltammogram_list(voltammogram_list, peak_info_list)

    # Area under the curve of all peaks:
    extracted_peak_params = [x['extracted_peak_parameters'] for x in voltammogram_list]
    pipeline.loop_integrate_peaks(extracted_peak_params, voltammograms_processed) 

    # Integrate baseline under peaks:
    pipeline.loop_integrate_peak_baselines(extracted_peak_params, voltammograms_processed) 

    # Average background:
    pipeline.loop_get_average_baseline(voltammogram_list)


                            
    # Tabulation of extracted peak parameters:
    labels                 = np.array([x['qualifiers']['Label'] for x in voltammogram_list])     # so we can differentiate between peak information for control, THC, or other solution types
    timestamps             = np.array([x['qualifiers']['Timestamp'] for x in voltammogram_list])   # tabulating the time stamps will allow us to analyze changes of peak parameters over time
    extracted_params_table = pipeline.build_analyte_data_table(labels, peak_info_list, timestamps)                                

    # Add 'Scan Order' column to the extracted parameters table:
    extracted_params_table = pipeline.add_scan_order(extracted_params_table, rank_column='Timestamp')


    # Save data table of various parameters extracted from the processed voltammograms:
    extracted_params_path = pipeline.make_subfolder(analysis_directory, 'Extracted Peak Parameters')
    extracted_params_table.to_csv(extracted_params_path + '\\' + 'extracted parameters.csv', index=False)

    all_extracted_params_tables.append(extracted_params_table) # this list is used later for batch analyses
    # Save individual dict entries in voltammogram_list, where each dict contains all of the info for a single scan
    sweep_info_path = pipeline.make_subfolder(analysis_directory, 'Sweep Info')
    filenames_no_suffix = [x[:-4] for x in voltammogram_files] # remove '.csv' from the filenames
    pipeline.save_sweep_info_to_files(voltammogram_list, voltammogram_files, sweep_info_path) # save/pickle each scan info dict 

    # CREATION OF THE MEASUREMENT DICTIONARY:
    # We now have a list of dicts. Each dict contains the original voltammogram, as a DataFrame, along with various other elements
    # that have been created while processing the original voltammogram (e.g. the extracted parameters like peak heights, potentials, etc)
    # This list of dicts will be added to a new dict called "measurement". This measurement dict will also contain the extracted 
    # parameters table. The measurement dictionary is meant to contain all of the information relevant to a single measurement (duh..
    # the name sort of creates itself doesn't it?). 
    # We'll also add the extracted parameters DataFrame to the measurement dictionary. While the individual sweep dictionaries within
    # voltammogram_list only contain the parameters from that sweep, it makes sense to tabulate all of those parameters and save them
    # in the overall measurement dictionary which contains all of the sweeps.

    # be pickled/saved such that all of the results of the analysis for a given voltammogram can be re-loaded later
    # without having to re-run the analysis pipeline over the data.
    voltammogram_list_path = pipeline.make_subfolder(analysis_directory, 'Pickled Voltammogram Info List')
    pipeline.pickle_voltammogram_info(voltammogram_list, voltammogram_list_path)


    # Create "measurement" dict which contains all scans from a given measurement (contained in voltammogram_list)
    # and the extracted parameters table (a pandas DataFrame):
    measurement = {
                    'voltammogram_list'    : voltammogram_list,
                    'extracted_parameters' : extracted_params_table
                }
    
    measurement_path = pipeline.make_subfolder(analysis_directory, 'Full Measurement Info')
    pipeline.pickle_measurement_info(measurement, measurement_path)

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################

    ############################# SECTION: DATA VISUALIZATION ########################################################################################
    ############################# SECTION: DATA VISUALIZATION ########################################################################################
    ############################# SECTION: DATA VISUALIZATION ########################################################################################
    ############################# SECTION: DATA VISUALIZATION ########################################################################################
    ############################# SECTION: DATA VISUALIZATION ########################################################################################
    
    # Make a Data Visualization folder in which to save all of the data visualization plots and data:
    visualization_dir = pipeline.make_subfolder(analysis_directory, 'Data Visualization')
    
    if data_vis_choices['pairplot'] == True:
        # Pair plot of extracted parameters where user can select columns and hue
        if i == 0:
            selected_columns, hue_column = pipeline.select_columns(extracted_params_table)

        extracted_params_pairplot     = pipeline.create_hue_pairplot(
                                                                        extracted_params_table,
                                                                        hue_column=hue_column,
                                                                        selected_columns=selected_columns
                                                                    )

        pipeline.save_plot(extracted_params_pairplot, visualization_dir, 'extracted_params_pairplot.png')

        # Plotting Peak Coordinates with hues for analyte and control:
        peak_coords_plot = pipeline.create_peak_coords_plot(extracted_params_table,
                                                            xlabel='Potential\n(V)', 
                                                            ylabel='Current\n(nA)', 
                                                            title='Peak-Coordinates Density')
        pipeline.save_plot(peak_coords_plot, visualization_dir, 'peak_coordinates_plot.png')




    # Plotting Peak Coordinates with Timestamp as a colour factor:
    # Create plot:
    if data_vis_choices['2D_peak_coords_drift'] == True:
        peak_coords_time_drift_fig, peak_coords_time_drift_ax = pipeline.labeled_colored_scatterplot(
                                                                                                        extracted_params_table['peak_potentials'].to_numpy(),
                                                                                                        extracted_params_table['peak_heights'].to_numpy(),
                                                                                                        extracted_params_table['Timestamp'].to_numpy(),
                                                                                                        extracted_params_table['Label'].to_numpy(),
                                                                                                        min_color_frac=0.1,
                                                                                                        max_color_frac=0.9,
                                                                                                        xlabel='Potential\n(V)',
                                                                                                        ylabel='Current\n(nA)',
                                                                                                        title='Time-Drift of Peak-Coordinates',
                                                                                                        zlabel='Timestamp\n(s)'
                                                                                                    )
        # save plot:
        pipeline.save_plot(peak_coords_time_drift_fig, visualization_dir, 'peak_coords_time_drift.png')


    if data_vis_choices['1D_peak_coords_drift'] == True:
        # Plotting current and potential against time:
        iE_vs_t_dual_fig, iE_vs_t_dual_ax = pipeline.plot_current_potential_vs_time(extracted_params_table)
        pipeline.save_plot(iE_vs_t_dual_fig, visualization_dir, 'peak_coords_dual_axis_vs_time.png')



    if data_vis_choices['overlay_tagged_sweeps'] == True:
        # Plotting all scans from same measurement on single plot with tagged peaks and time
        # as a colour-factor:
        overlay_scans_figures, overlay_scans_axes, unique_labels = pipeline.overlay_scans(voltammogram_list)

        num_unique_labels = len(unique_labels)
        for i in range(num_unique_labels):
            pipeline.save_plot(overlay_scans_figures[i], 
                            visualization_dir, 
                            '{}_overlay_scans_tagged_peaks.png'.format(unique_labels[i])) 
        



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

############################# SECTION: BATCH ANALYSIS SPECIFICS ############################################################################################
############################# SECTION: BATCH ANALYSIS SPECIFICS ############################################################################################
############################# SECTION: BATCH ANALYSIS SPECIFICS ############################################################################################
############################# SECTION: BATCH ANALYSIS SPECIFICS ############################################################################################
############################# SECTION: BATCH ANALYSIS SPECIFICS ############################################################################################
    
if analysis_mode == 'batch':
    batch_analysis_folder         = pipeline.make_subfolder(parent_folderpath, 'Batch Analysis')
    # Add tags to extracted parameter tables, before concatenating them into a batch, such that
    # they can be differentiated in the concatenated table:
    pipeline.add_measurement_tags(all_measurements, all_extracted_params_tables)
    pooled_extracted_params_table = pd.concat(all_extracted_params_tables, ignore_index=True)
    pooled_params_folder          = pipeline.make_subfolder(batch_analysis_folder, 'pooled_extracted_params')

    pooled_extracted_params_table.to_csv(pooled_params_folder + '\\' + 'pooled_extracted_params.csv', index=False)



 