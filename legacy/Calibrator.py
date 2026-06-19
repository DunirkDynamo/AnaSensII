import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from pandastable import Table
import os
import pickle

class VoltammogramCalibrator:
    def __init__(self, root):
        self.root = root
        self.root.title("Voltammogram Calibrator")
        self.data_format = None
        self.dataframe = None

        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create format selector
        self.create_data_format_selector()

        self.match_filters = []  # List to store selected filter columns
        self.filter_dropdown = None  # For reference to the dropdown widget
        self.calibration_method = None  # add this if missing

        self.previous_filters = []
        self.previous_result_indices = []
        self.previous_result_table = None




    def create_data_format_selector(self):
        """
        Purpose:
            Builds the initial UI section that allows the user to choose between
            two different calibration modes:
            - Calibrate Data Table (a single CSV file)
            - Calibrate Measurement Objects (multiple pickled measurement folders)
    
        Inputs:
            - None directly. It builds interface components.
    
        Effect:
            - Adds a labeled dropdown menu and a confirm button to the GUI.
            - When a format is confirmed, the selection is passed to set_data_format().
        """
        # Create a labeled frame for the data format section
        self.format_frame = tk.LabelFrame(
            self.main_frame, text="Data Format", font=('Arial', 10, 'bold')
        )
        self.format_frame.pack(fill=tk.X, pady=5)
    
        # Instructional label
        tk.Label(self.format_frame, text="Select Data Format:").pack(side=tk.LEFT, padx=5)
    
        # Dropdown menu for format selection
        self.format_var = tk.StringVar()
        format_options = ['Calibrate Data Table', 'Calibrate Measurement Objects']
        format_dropdown = ttk.Combobox(
            self.format_frame, textvariable=self.format_var,
            values=format_options, state='readonly', width=25
        )
        format_dropdown.pack(side=tk.LEFT, padx=5)
        format_dropdown.current(0)  # Default selection
    
        # Confirm button to finalize the selection
        confirm_btn = tk.Button(self.format_frame, text="Confirm", command=self.set_data_format)
        confirm_btn.pack(side=tk.LEFT, padx=10)
    
        # Label to show the current format status
        self.format_status = tk.Label(self.format_frame, text="No format selected")
        self.format_status.pack(side=tk.LEFT, padx=10)


    def set_data_format(self):
        """
        Purpose:
            Captures the selected calibration format from the dropdown and triggers
            the appropriate data loading function based on the user’s selection.
    
        Inputs:
            - Reads from self.format_var (UI selection made by user)
    
        Effect:
            - Updates self.data_format with the selected value
            - Removes the format selector UI from view
            - Calls either load_data_table() or load_measurement_objects()
        """
        # Get the selected value from the dropdown
        self.data_format = self.format_var.get()
    
        # If nothing selected, warn the user
        if not self.data_format:
            messagebox.showwarning("Warning", "Please select a data format")
            return
    
        # Update UI to reflect the chosen format
        self.format_status.config(text=f"Selected format: {self.data_format}")
    
        # Hide the format selector after confirming
        self.format_frame.pack_forget()
    
        # Route to the appropriate data loading function
        if self.data_format == 'Calibrate Data Table':
            self.load_data_table()
        elif self.data_format == 'Calibrate Measurement Objects':
            self.load_measurement_objects()



    def load_data_table(self):
        """Load and preview a CSV data table"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
    
        if not file_path:
            return
    
        try:
            self.dataframe = pd.read_csv(file_path)
    
            # Clear previous table frame if it exists
            if hasattr(self, 'table_frame') and self.table_frame.winfo_exists():
                self.table_frame.destroy()
    
            self.table_frame = tk.LabelFrame(self.main_frame, text="Data Table Preview (10x10)",
                                             font=('Arial', 10, 'bold'))
            self.table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
            preview_df = self.dataframe
    
            self.table = Table(
                self.table_frame,
                dataframe=preview_df,
                showtoolbar=False,
                showstatusbar=True,
                width=700,
                height=300
            )
            self.table.show()
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")

        self.create_calibration_selector()
    


    def create_calibration_selector(self):
        """Create the calibration method and label selector section"""
        if hasattr(self, 'calib_frame') and self.calib_frame.winfo_exists():
            self.calib_frame.destroy()
    
        self.calib_frame = tk.LabelFrame(self.main_frame, text="Calibration", font=('Arial', 10, 'bold'))
        self.calib_frame.pack(fill=tk.X, pady=5)
    
        # --- Row 1: Calibration Method ---
        method_row = tk.Frame(self.calib_frame)
        method_row.pack(fill=tk.X, pady=2)
    
        tk.Label(method_row, text="Calibration Method:").pack(side=tk.LEFT, padx=5)
    
        self.calib_var = tk.StringVar()
        calib_options = ['subcal', 'divcal']
        calib_dropdown = ttk.Combobox(method_row, textvariable=self.calib_var,
                                      values=calib_options, state='readonly', width=10)
        calib_dropdown.pack(side=tk.LEFT, padx=5)
        calib_dropdown.current(0)
    
        tk.Button(method_row, text="Confirm Method", command=self.confirm_calibration_method).pack(side=tk.LEFT, padx=5)
    
        self.calib_status = tk.Label(method_row, text="No method selected")
        self.calib_status.pack(side=tk.LEFT, padx=5)
    
        # --- Row 2: Analyte and Control Selection ---
        label_row = tk.Frame(self.calib_frame)
        label_row.pack(fill=tk.X, pady=2)
    
        unique_labels = sorted(self.dataframe['Label'].unique().tolist()) if 'Label' in self.dataframe.columns else []
    
        tk.Label(label_row, text="Analyte:").pack(side=tk.LEFT, padx=5)
        self.analyte_var = tk.StringVar()
        self.analyte_menu = ttk.Combobox(label_row, textvariable=self.analyte_var,
                                         values=unique_labels, state='readonly', width=10)
        self.analyte_menu.pack(side=tk.LEFT)
        tk.Button(label_row, text="Set Analyte", command=self.set_analyte).pack(side=tk.LEFT, padx=2)
    
        self.analyte_status = tk.Label(label_row, text="None")
        self.analyte_status.pack(side=tk.LEFT, padx=5)
    
        tk.Label(label_row, text="Control:").pack(side=tk.LEFT, padx=5)
        self.control_var = tk.StringVar()
        self.control_menu = ttk.Combobox(label_row, textvariable=self.control_var,
                                         values=unique_labels, state='readonly', width=10)
        self.control_menu.pack(side=tk.LEFT)
        tk.Button(label_row, text="Set Control", command=self.set_control).pack(side=tk.LEFT, padx=2)
    
        self.control_status = tk.Label(label_row, text="None")
        self.control_status.pack(side=tk.LEFT, padx=5)
    
        # --- Row 3: Filter Controls ---
        filter_row = tk.Frame(self.calib_frame)
        filter_row.pack(fill=tk.X, pady=2)
    
        tk.Button(filter_row, text="Add Match Filter", command=self.add_match_filter).pack(side=tk.LEFT, padx=10)
        tk.Button(filter_row, text="Clear Filters", command=self.clear_match_filters).pack(side=tk.LEFT, padx=10)
    
        # Label to show added filters
        self.filter_label = tk.Label(filter_row, text="Filters: None")
        self.filter_label.pack(side=tk.LEFT, padx=10)
    
        # --- Row 4: Run Calibration ---
        action_row = tk.Frame(self.calib_frame)
        action_row.pack(fill=tk.X, pady=5)
    
        #tk.Button(action_row, text="Run Calibration", command=self.run_calibration).pack(padx=10)

        tk.Button(action_row, text="Run Calibration", command=self.run_calibration).pack(side=tk.LEFT, padx=10)
        self.export_btn = tk.Button(action_row, text="Export to CSV", command=self.export_calibration_result, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=10)


    def set_analyte(self):
        """Set analyte label"""
        analyte = self.analyte_var.get()
        if analyte:
            self.analyte_status.config(text=f"{analyte}")
        else:
            messagebox.showwarning("Warning", "Please select an analyte label")

    def set_control(self):
        """Set control label"""
        control = self.control_var.get()
        if control:
            self.control_status.config(text=f"{control}")
        else:
            messagebox.showwarning("Warning", "Please select a control label")

    def add_match_filter(self):
        """
        Displays a dropdown menu to select a match filter column,
        adds the selected column to the list, and updates the visible list of filters.
        """
        if not hasattr(self, 'filter_frame'):
            self.filter_frame = tk.LabelFrame(self.calib_frame, text="Match Filters", font=('Arial', 10, 'bold'))
            self.filter_frame.pack(fill=tk.X, pady=5)
    
        # Remove existing dropdown if present
        if self.filter_dropdown:
            self.filter_dropdown.destroy()
    
        # Dropdown
        filter_var = tk.StringVar()
        filter_options = list(self.dataframe.columns)
        self.filter_dropdown = ttk.Combobox(self.filter_frame, textvariable=filter_var,
                                            values=filter_options, state='readonly', width=20)
        self.filter_dropdown.pack(side=tk.LEFT, padx=5)
    
        # Confirm button
        def confirm_filter():
            selected_filter = filter_var.get()
            if selected_filter and selected_filter not in self.match_filters:
                self.match_filters.append(selected_filter)
                messagebox.showinfo("Filter Added", f"Filter '{selected_filter}' added.")
                self.update_filter_display()
            self.filter_dropdown.destroy()
            self.filter_dropdown = None
    
        confirm_btn = tk.Button(self.filter_frame, text="Confirm Filter", command=confirm_filter)
        confirm_btn.pack(side=tk.LEFT, padx=5)
    
        # Filter display label
        if not hasattr(self, 'filter_list_label'):
            self.filter_list_label = tk.Label(self.filter_frame, text="")
            self.filter_list_label.pack(side=tk.LEFT, padx=10)



    def clear_match_filters(self):
        """
        Clears all selected match filters and resets the match filter UI frame.
        """
        self.match_filters = []
    
        if hasattr(self, 'filter_frame') and self.filter_frame.winfo_exists():
            for widget in self.filter_frame.winfo_children():
                widget.destroy()
            self.filter_frame.destroy()
            del self.filter_frame
    
        if hasattr(self, 'filter_list_label'):
            self.filter_list_label.destroy()
            del self.filter_list_label
    
        messagebox.showinfo("Filters Cleared", "All match filters have been cleared.")

    


    
    def calibrate_data_table(self, method):
        """
        Purpose:
        Performs calibration (subcal or divcal) on the loaded DataFrame using match filters to pair analyte and control rows.
    
        Input:
        - method (str): Calibration method to use ('subcal' or 'divcal').
    
        Output:
        None
        """
        if self.dataframe is None:
            messagebox.showwarning("Warning", "No data table loaded.")
            return
    
        if not hasattr(self, 'analyte_label') or not hasattr(self, 'control_label'):
            messagebox.showwarning("Warning", "Please select Analyte and Control labels.")
            return
    
        analyte = self.analyte_label
        control = self.control_label
    
        try:
            results = []
            for idx, row in self.dataframe.iterrows():
                if row['Label'] != analyte:
                    continue
    
                # Build filter conditions
                filter_conditions = (self.dataframe['Label'] == control)
                for filt in self.match_filters:
                    filter_conditions &= (self.dataframe[filt] == row[filt])
    
                matching_rows = self.dataframe[filter_conditions]
    
                if len(matching_rows) != 1:
                    results.append(float('nan'))
                    continue
    
                control_val = matching_rows.iloc[0]['peak_heights']
                analyte_val = row['peak_heights']
    
                if method == 'subcal':
                    result = analyte_val - control_val
                elif method == 'divcal':
                    result = analyte_val / control_val if control_val != 0 else float('nan')
                else:
                    result = float('nan')
    
                results.append(result)
    
            # Create a new column only if it doesn't already exist
            if method in self.dataframe.columns:
                messagebox.showinfo("Already Calculated", f"{method} has already been calculated.")
            else:
                # Mark all rows as NaN initially
                self.dataframe[method] = float('nan')
                # Assign calculated results to the appropriate analyte rows
                analyte_indices = self.dataframe.index[self.dataframe['Label'] == analyte]
                self.dataframe.loc[analyte_indices, method] = results
                self.display_calibrated_table()
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calibrate data table: {str(e)}")

    
    def run_calibration(self):
        """
        Executes the selected calibration method using Analyte, Control,
        and selected match filters. If the filters haven't changed, adds the new
        calibration to the existing result table; otherwise creates a new one.
        """
        if self.dataframe is None:
            messagebox.showerror("Error", "No dataframe loaded.")
            return
    
        self.calibration_method = self.calib_var.get()
        if not self.calibration_method:
            messagebox.showerror("Error", "No calibration method selected.")
            return
    
        self.calib_status.config(text=f"Selected method: {self.calibration_method}")
    
        if not hasattr(self, 'analyte_var') or not hasattr(self, 'control_var'):
            messagebox.showerror("Error", "Please select both Analyte and Control labels.")
            return
    
        self.analyte_label = self.analyte_var.get()
        self.control_label = self.control_var.get()
    
        if not self.analyte_label or not self.control_label:
            messagebox.showerror("Error", "Please select both Analyte and Control labels.")
            return
    
        try:
            df = self.dataframe.copy()
            analyte_rows = df[df["Label"] == self.analyte_label].copy()
            control_rows = df[df["Label"] == self.control_label].copy()
    
            results = []
            used_indices = []
            analyte_heights = []
            control_heights = []
            analyte_labels = []
            control_labels = []
    
            for idx, analyte_row in analyte_rows.iterrows():
                if self.match_filters:
                    match_conditions = (control_rows[self.match_filters] == analyte_row[self.match_filters].values).all(axis=1)
                else:
                    match_conditions = pd.Series([True] * len(control_rows), index=control_rows.index)
    
                matching_controls = control_rows[match_conditions]
    
                if len(matching_controls) == 1:
                    control_value = matching_controls.iloc[0]["peak_heights"]
                    analyte_value = analyte_row["peak_heights"]
    
                    if self.calibration_method == "subcal":
                        result = analyte_value - control_value
                    else:  # divcal
                        result = analyte_value / control_value if control_value != 0 else float('nan')
                else:
                    result = float('nan')
                    control_value = float('nan')
                    analyte_value = analyte_row["peak_heights"]
    
                results.append(result)
                used_indices.append(idx)
                control_heights.append(control_value)
                analyte_heights.append(analyte_value)
                analyte_labels.append(self.analyte_label)
                control_labels.append(self.control_label)
    
            colname = self.calibration_method
    
            if colname in df.columns and self.match_filters == getattr(self, 'previous_filters', []):
                messagebox.showinfo("Info", f"{colname} has already been run with the current filters.")
                return
    
            df.loc[used_indices, colname] = results
            df.loc[used_indices, f"peak_heights_{self.analyte_label}"] = analyte_heights
            df.loc[used_indices, f"peak_heights_{self.control_label}"] = control_heights
            df.loc[used_indices, "Analyte"] = analyte_labels
            df.loc[used_indices, "Control"] = control_labels
    
            self.dataframe = df
    
            display_cols = self.match_filters + [
                "Timestamp",
                "Analyte",
                "Control",
                f"peak_heights_{self.analyte_label}",
                f"peak_heights_{self.control_label}",
                colname,
            ]
    
            filters_changed = self.match_filters != getattr(self, 'previous_filters', [])
            if filters_changed or not hasattr(self, 'previous_result_table') or self.previous_result_table is None:
                filtered_df = df.loc[used_indices, display_cols].copy()
                self.display_calibration_result(filtered_df)
                self.export_btn.config(state=tk.NORMAL)  
                self.previous_filters = self.match_filters.copy()
                self.previous_result_indices = used_indices.copy()
                self.previous_result_table = filtered_df
            else:
                # Add new result to existing displayed DataFrame
                self.previous_result_table[colname] = results
                self.display_calibration_result(self.previous_result_table)
                self.export_btn.config(state=tk.NORMAL)
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run calibration: {str(e)}")

    def display_calibration_result(self, result_df):
        """
        Displays the calibration result dataframe in a scrollable table within a dedicated frame.
        Also provides an export-to-CSV button.
        """
        if hasattr(self, 'calibration_result_frame') and self.calibration_result_frame.winfo_exists():
            self.calibration_result_frame.destroy()
    
        self.calibration_result_frame = tk.LabelFrame(self.main_frame, text="Calibration Result", font=('Arial', 10, 'bold'))
        self.calibration_result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
        # Scrollbars
        xscrollbar = tk.Scrollbar(self.calibration_result_frame, orient=tk.HORIZONTAL)
        yscrollbar = tk.Scrollbar(self.calibration_result_frame)
    
        # Table display
        result_table = Table(self.calibration_result_frame, dataframe=result_df,
                             showtoolbar=False, showstatusbar=False, width=600, height=300)
        result_table.show()
    
        xscrollbar.config(command=result_table.xview)
        yscrollbar.config(command=result_table.yview)
    
        result_table.config(xscrollcommand=xscrollbar.set,
                            yscrollcommand=yscrollbar.set)
    
        #xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        #yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
        # Store latest result for exporting
        self.latest_result_df = result_df.copy()
    
        # Export button
        #export_btn = tk.Button(self.calibration_result_frame, text="Export to CSV", command=self.export_calibration_result)
        #export_btn.pack(side=tk.BOTTOM, pady=5)

        # Store latest result for exporting and enable export button
        self.latest_result_df = result_df.copy()
        self.export_btn.config(state=tk.NORMAL)


    
    def set_calibration_method(self):
        """Sets the calibration method based on user selection"""
        method = self.calib_var.get()
        
        if not method:
            messagebox.showwarning("Warning", "Please select a calibration method")
            return
        
        self.calibration_method = method  # Update the calibration method
        self.calib_status.config(text=f"Selected method: {method}")  # Update the status text


    def confirm_calibration_method(self):
        """Stores and displays the selected calibration method without running calibration"""
        self.calibration_method = self.calib_var.get()
        if not self.calibration_method:
            messagebox.showwarning("Warning", "Please select a calibration method")
        else:
            self.calib_status.config(text=f"Selected method: {self.calibration_method}")

    def update_filter_display(self):
        """Update the text label showing selected match filters."""
        if hasattr(self, 'filter_list_label'):
            self.filter_list_label.config(text="Filters: " + ", ".join(self.match_filters))

    def export_calibration_result(self):
        """
        Allows user to select a folder and filename to export the latest calibration result as CSV.
        """
        if not hasattr(self, 'latest_result_df') or self.latest_result_df.empty:
            self.export_btn.config(state=tk.DISABLED)
            messagebox.showerror("Error", "No calibration result to export.")
            self.export_btn.config(state=tk.DISABLED)
            return

        
        folder = filedialog.askdirectory(title="Select Folder to Save CSV")
        if not folder:
            return
    
        file_name = tk.simpledialog.askstring("Save As", "Enter the name for the CSV file (no extension):")
        if not file_name:
            return
    
        file_path = f"{folder}/{file_name}.csv"
        try:
            self.latest_result_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"File saved to:\n{file_path}")
        except Exception as e:
            self.export_btn.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Failed to export file:\n{str(e)}")




    def load_measurement_objects(self):
        """
        Loads multiple measurement objects from pickled dicts in the following structure:
            parent_dir/
                measurement1/Full Measurement Info/measurement_info.pkl
                measurement2/Full Measurement Info/measurement_info.pkl
                ...
        Each pickle file is expected to be a dict containing an 'extracted_parameters' DataFrame.
        Stacks all data and displays it with a 'Measurement' column.
        """
        parent_dir = filedialog.askdirectory(title="Select Parent Folder with Measurement Subdirectories")
        if not parent_dir:
            return
    
        self.measurement_objects = []  # Reset list
        self.last_loaded_parent_dir = parent_dir  # Store for later saving


        for subdir_name in os.listdir(parent_dir):
            subdir_path = os.path.join(parent_dir, subdir_name)
            if not os.path.isdir(subdir_path):
                continue
    
            # Construct path to the pickle file
            pkl_path = os.path.join(subdir_path, "Full Measurement Info", "measurement_info.pkl")
            if os.path.isfile(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        measurement_data = pickle.load(f)
    
                    if 'extracted_parameters' in measurement_data:
                        df = measurement_data['extracted_parameters']
                        if isinstance(df, pd.DataFrame):
                            self.measurement_objects.append({
                                'extracted_parameters': df,
                                'measurement_name': subdir_name
                            })
                        else:
                            messagebox.showwarning("Invalid Format", f"'extracted_parameters' in {subdir_name} is not a DataFrame.")
                    else:
                        messagebox.showwarning("Missing Key", f"'extracted_parameters' not found in {subdir_name}.")
                except Exception as e:
                    messagebox.showwarning("Load Failed", f"Could not load pickle for '{subdir_name}': {str(e)}")
    
        if not self.measurement_objects:
            messagebox.showwarning("No Data", "No valid measurement objects were found.")
            return
    
        # Stack DataFrames and add Measurement column
        combined_data = []
        for obj in self.measurement_objects:
            df = obj['extracted_parameters'].copy()
            df['Measurement'] = obj['measurement_name']
            combined_data.append(df)
    
        self.stacked_measurement_df = pd.concat(combined_data, ignore_index=True)
    
        # Move 'Measurement' to the first column
        cols = self.stacked_measurement_df.columns.tolist()
        if "Measurement" in cols:
            cols.insert(0, cols.pop(cols.index("Measurement")))
            self.stacked_measurement_df = self.stacked_measurement_df[cols]
    
        # Display the stacked DataFrame in a table
        if hasattr(self, 'table_frame') and self.table_frame.winfo_exists():
            self.table_frame.destroy()
    
        self.table_frame = tk.LabelFrame(self.main_frame, text="Stacked Measurement Data",
                                         font=('Arial', 10, 'bold'))
        self.table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
        self.table = Table(
            self.table_frame,
            dataframe=self.stacked_measurement_df,
            showtoolbar=False,
            showstatusbar=True,
            width=700,
            height=300
        )
        self.table.show()
    
        messagebox.showinfo("Measurements Loaded", f"Loaded and stacked {len(self.measurement_objects)} measurement objects.")
        self.create_calibration_selector_for_measurements()


    def create_calibration_selector_for_measurements(self):
        """Creates the calibration control panel for measurement object mode."""
        if hasattr(self, 'calib_frame') and self.calib_frame.winfo_exists():
            self.calib_frame.destroy()
    
        self.calib_frame = tk.LabelFrame(self.main_frame, text="Calibration for Measurement Objects",
                                         font=('Arial', 10, 'bold'))
        self.calib_frame.pack(fill=tk.X, pady=5)
    
        # Calibration method
        tk.Label(self.calib_frame, text="Calibration Method:").pack(side=tk.LEFT, padx=5)
        self.calib_var = tk.StringVar()
        calib_options = ['subcal', 'divcal']
        calib_dropdown = ttk.Combobox(self.calib_frame, textvariable=self.calib_var,
                                      values=calib_options, state='readonly', width=10)
        calib_dropdown.pack(side=tk.LEFT, padx=5)
        calib_dropdown.current(0)
    
        tk.Button(self.calib_frame, text="Confirm Method", command=self.confirm_calibration_method).pack(side=tk.LEFT, padx=5)
    
        self.calib_status = tk.Label(self.calib_frame, text="No method selected")
        self.calib_status.pack(side=tk.LEFT, padx=10)
    
        # Collect unique labels across all DataFrames
        label_set = set()
        for obj in self.measurement_objects:
            df = obj['extracted_parameters']
            if 'Label' in df.columns:
                label_set.update(df['Label'].dropna().unique().tolist())
    
        unique_labels = sorted(label_set)
    
        # Analyte selector
        tk.Label(self.calib_frame, text="Analyte:").pack(side=tk.LEFT, padx=5)
        self.analyte_var = tk.StringVar()
        self.analyte_menu = ttk.Combobox(self.calib_frame, textvariable=self.analyte_var,
                                         values=unique_labels, state='readonly', width=10)
        self.analyte_menu.pack(side=tk.LEFT)
        tk.Button(self.calib_frame, text="Set Analyte", command=self.set_analyte).pack(side=tk.LEFT, padx=2)
        self.analyte_status = tk.Label(self.calib_frame, text="None")
        self.analyte_status.pack(side=tk.LEFT, padx=5)
    
        # Control selector
        tk.Label(self.calib_frame, text="Control:").pack(side=tk.LEFT, padx=5)
        self.control_var = tk.StringVar()
        self.control_menu = ttk.Combobox(self.calib_frame, textvariable=self.control_var,
                                         values=unique_labels, state='readonly', width=10)
        self.control_menu.pack(side=tk.LEFT)
        tk.Button(self.calib_frame, text="Set Control", command=self.set_control).pack(side=tk.LEFT, padx=2)
        self.control_status = tk.Label(self.calib_frame, text="None")
        self.control_status.pack(side=tk.LEFT, padx=5)
    
        # Run button
        tk.Button(self.calib_frame, text="Run Calibration", command=self.calibrate_measurement_objects).pack(side=tk.LEFT, padx=10)

        tk.Button(self.calib_frame, text="Finalize & Save All", command=self.finalize_calibration_and_save).pack(side=tk.LEFT, padx=10)

        
        
    def calibrate_measurement_objects(self):
        method = self.calib_var.get()
        if not method:
            messagebox.showwarning("Warning", "Please select a calibration method")
            return
    
        analyte_label = self.analyte_var.get()
        control_label = self.control_var.get()
        if not analyte_label or not control_label:
            messagebox.showwarning("Warning", "Please select both Analyte and Control labels")
            return
    
        calibrated_count = 0
        debug_messages = []
        all_calibration_results = []
    
        for obj in self.measurement_objects:
            df = obj['extracted_parameters']
            obj_name = obj['measurement_name']
    
            # Validate DataFrame structure
            missing_cols = [col for col in ['Label', 'Sweep', 'peak_heights'] if col not in df.columns]
            if missing_cols:
                debug_messages.append(f"⚠️ {obj_name}: Missing columns {missing_cols}")
                continue
    
            # Verify labels exist
            if analyte_label not in df['Label'].values:
                debug_messages.append(f"⚠️ {obj_name}: Missing analyte '{analyte_label}'")
                continue
            if control_label not in df['Label'].values:
                debug_messages.append(f"⚠️ {obj_name}: Missing control '{control_label}'")
                continue
    
            # Get filtered rows
            analyte_rows = df[df['Label'] == analyte_label]
            control_rows = df[df['Label'] == control_label]
            calibration_entries = []
    
            # Process analyte-control pairs
            for _, analyte_row in analyte_rows.iterrows():
                # Build filter conditions
                filter_conditions = (control_rows['Sweep'] == analyte_row['Sweep'])
                for filt in self.match_filters:
                    if filt in df.columns:
                        filter_conditions &= (control_rows[filt] == analyte_row[filt])
    
                matching_controls = control_rows[filter_conditions]
    
                if len(matching_controls) != 1:
                    debug_messages.append(f"⚠️ {obj_name}: No 1:1 match for Sweep {analyte_row['Sweep']}")
                    continue
    
                control_row = matching_controls.iloc[0]
    
                # Create new entry mirroring calibrate_data_table format
                entry = {
                    'Measurement': obj_name,
                    'Sweep': analyte_row['Sweep'],
                    'Analyte': analyte_label,
                    'Control': control_label,
                    'peak_heights_analyte': analyte_row['peak_heights'],
                    'peak_heights_control': control_row['peak_heights'],
                    method: None
                }
    
                # Add match filters to columns
                for filt in self.match_filters:
                    entry[filt] = analyte_row[filt]
    
                # Calculate calibration result
                try:
                    a_val = float(entry['peak_heights_analyte'])
                    c_val = float(entry['peak_heights_control'])
                    
                    if method == 'subcal':
                        entry[method] = a_val - c_val
                    else:  # divcal
                        entry[method] = a_val / c_val if c_val != 0 else float('nan')
                        
                    calibrated_count += 1
                except (TypeError, ValueError) as e:
                    entry[method] = float('nan')
                    debug_messages.append(f"⚠️ {obj_name}: Invalid values (Sweep {analyte_row['Sweep']}): {str(e)}")
    
                calibration_entries.append(entry)
    
            # Store results
            if calibration_entries:
                obj['calibration_table'] = pd.DataFrame(calibration_entries)
                all_calibration_results.extend(calibration_entries)
                debug_messages.append(f"✅ {obj_name}: Created {len(calibration_entries)} calibration pairs")
    
        # Create unified table
        # Create unified table
        if all_calibration_results:
            self.stacked_calibration_table = pd.DataFrame(all_calibration_results)
        
            # Ensure 'Sweep' is included in the column order
            col_order = ['Measurement'] + self.match_filters + [
                'Sweep', 'Analyte', 'Control',
                'peak_heights_analyte', 'peak_heights_control',
                method
            ]
            self.stacked_calibration_table = self.stacked_calibration_table.reindex(columns=[col for col in col_order if col in self.stacked_calibration_table.columns])

    
        # Show debug info
        print("\n".join(debug_messages))
        messagebox.showinfo(
            "Calibration Complete",
            f"Created {len(all_calibration_results)} calibration pairs\n"
            f"Across {len(self.measurement_objects)} measurements\n"
            f"See console for detailed warnings"
        )
    
        # Display results if available
        if hasattr(self, 'stacked_calibration_table'):
            self.display_calibration_result(self.stacked_calibration_table)


    def finalize_calibration_and_save(self):
        """
        Purpose:
            Finalizes calibration for each measurement object:
            - Ensures each object has a full-format calibration_table
            - Saves each object back to its original pickle
            - Prompts user to export the existing stacked calibration table to CSV
    
        Inputs:
            - self.measurement_objects
            - self.match_filters, self.calib_var, self.analyte_var, self.control_var
            - self.last_loaded_parent_dir
    
        Output:
            - Saves each calibrated object
            - Prompts and saves self.stacked_calibration_table to CSV
        """
        if not hasattr(self, 'measurement_objects') or not self.measurement_objects:
            messagebox.showwarning("No Data", "No measurement objects found.")
            return
    
        method = self.calib_var.get()
        analyte_label = self.analyte_var.get()
        control_label = self.control_var.get()
    
        if not method or not analyte_label or not control_label:
            messagebox.showwarning("Missing Info", "Please ensure method, analyte, and control are selected.")
            return
    
        stacked_results = []
    
        for obj in self.measurement_objects:
            measurement_name = obj.get('measurement_name')
    
            if 'calibration_table' not in obj:
                continue
    
            table = obj['calibration_table'].copy()
            if table.empty or method not in table.columns:
                continue
    
            table['Measurement'] = measurement_name
            table['Analyte'] = analyte_label
            table['Control'] = control_label
    
            col_order = ['Measurement'] + self.match_filters + [
                'Analyte', 'Control',
                'peak_heights_analyte', 'peak_heights_control',
                method
            ]
            table = table[[col for col in col_order if col in table.columns]]
    
            obj['calibration_table'] = table.copy()
    
            original_path = os.path.join(
                self.last_loaded_parent_dir,
                measurement_name,
                "Full Measurement Info",
                "measurement_info.pkl"
            )
            try:
                with open(original_path, "wb") as f:
                    pickle.dump(obj, f)
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save {measurement_name}:\n{str(e)}")
                continue
    
            stacked_results.append(table)
    
        if not stacked_results:
            messagebox.showinfo("No Output", "No calibration tables were finalized.")
            return
    
        # Update the in-memory stack for export, but don't display again
        full_stack = pd.concat(stacked_results, ignore_index=True)
        self.stacked_calibration_table = full_stack
        self.latest_export_df = full_stack.copy()
    
        # Prompt user for save location
        save_dir = filedialog.askdirectory(title="Select Folder to Save Stacked Calibration Table")
        if not save_dir:
            return  # User canceled
    
        filename = tk.simpledialog.askstring("Save As", "Enter filename (no extension):")
        if not filename:
            return
    
        save_path = os.path.join(save_dir, f"{filename}.csv")
        try:
            self.latest_export_df.to_csv(save_path, index=False)
            messagebox.showinfo("Export Successful", f"Stacked calibration table saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file:\n{str(e)}")
    
        
    



    def export_stacked_calibration_table(self):
        """Prompts the user to export the stacked calibration table as a CSV."""
        if not hasattr(self, 'latest_export_df') or self.latest_export_df.empty:
            messagebox.showerror("Error", "No calibration result to export.")
            return
    
        folder = filedialog.askdirectory(title="Select Folder to Save CSV")
        if not folder:
            return
    
        file_name = tk.simpledialog.askstring("Save As", "Enter the name for the CSV file (no extension):")
        if not file_name:
            return
    
        file_path = os.path.join(folder, f"{file_name}.csv")
        try:
            self.latest_export_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"File saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export file:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = VoltammogramCalibrator(root)
    root.mainloop()
