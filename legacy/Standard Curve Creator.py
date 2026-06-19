import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import Ekidna as ek
import pickle
import os

class DataFrameVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("DataFrame Visualizer with Ekidna")
        self.data = None
        self.current_fit = None
        self.alpha = 0.05  # Default confidence level
        self.x_units = ""  # Track x-axis units
        self.y_units = ""  # Track y-axis units
        
        # Model options from Ekidna
        self.MODELS = {
            "LLSR": ek.linear_std_curve,
            "PLSR": ek.parabola_std_curve,
            "LQR": ek.linear_QR,
            "PQR": ek.parabolic_QR,
            "PowQR": ek.power_std_curve
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize all UI components"""
        # Create main frames
        self.nav_frame = ttk.LabelFrame(self.master, text="Navigation", padding=10)
        self.nav_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.display_frame = ttk.LabelFrame(self.master, text="Data Display", padding=10)
        self.display_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.plot_frame = ttk.LabelFrame(self.master, text="Visualization", padding=10)
        self.plot_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Data display with scrollbars
        self.text_container = ttk.Frame(self.display_frame)
        self.text_container.grid(row=0, column=0, sticky="nsew")
        
        self.data_xscroll = ttk.Scrollbar(self.display_frame, orient="horizontal")
        self.data_xscroll.grid(row=1, column=0, sticky="ew")
        
        self.data_yscroll = ttk.Scrollbar(self.display_frame)
        self.data_yscroll.grid(row=0, column=1, sticky="ns")
        
        self.data_text = tk.Text(
            self.text_container, 
            height=15, 
            width=40,
            wrap=tk.NONE,
            xscrollcommand=self.data_xscroll.set,
            yscrollcommand=self.data_yscroll.set
        )
        self.data_text.grid(row=0, column=0, sticky="nsew")
        
        self.data_xscroll.config(command=self.data_text.xview)
        self.data_yscroll.config(command=self.data_text.yview)
        
        self.text_container.rowconfigure(0, weight=1)
        self.text_container.columnconfigure(0, weight=1)
        
        # Plot controls
        self.control_frame = ttk.Frame(self.nav_frame)
        self.control_frame.grid(row=1, column=0, pady=10, sticky="ew")
        
        # Axis selection with units
        ttk.Label(self.control_frame, text="x-axis:").grid(row=0, column=0, padx=5)
        self.x_var = tk.StringVar()
        self.x_dropdown = ttk.Combobox(self.control_frame, textvariable=self.x_var, state="readonly")
        self.x_dropdown.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.control_frame, text="x-units:").grid(row=0, column=2, padx=5)
        self.x_units_var = tk.StringVar()
        self.x_units_entry = ttk.Entry(self.control_frame, textvariable=self.x_units_var, width=8)
        self.x_units_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(self.control_frame, text="y-axis:").grid(row=1, column=0, padx=5)
        self.y_var = tk.StringVar()
        self.y_dropdown = ttk.Combobox(self.control_frame, textvariable=self.y_var, state="readonly")
        self.y_dropdown.grid(row=1, column=1, padx=5)
        
        ttk.Label(self.control_frame, text="y-units:").grid(row=1, column=2, padx=5)
        self.y_units_var = tk.StringVar()
        self.y_units_entry = ttk.Entry(self.control_frame, textvariable=self.y_units_var, width=8)
        self.y_units_entry.grid(row=1, column=3, padx=5)
        
        # Model selection
        ttk.Label(self.control_frame, text="Model:").grid(row=2, column=0, padx=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            self.control_frame,
            textvariable=self.model_var,
            values=list(self.MODELS.keys()),
            state="readonly"
        )
        self.model_dropdown.grid(row=2, column=1, padx=5)
        self.model_dropdown.current(0)
        
        # Confidence level control
        ttk.Label(self.control_frame, text="Confidence:").grid(row=3, column=0, padx=5)
        self.alpha_var = tk.StringVar(value="0.05")
        self.alpha_entry = ttk.Entry(self.control_frame, textvariable=self.alpha_var, width=8)
        self.alpha_entry.grid(row=3, column=1, padx=5)
        
        # Filter controls
        ttk.Label(self.control_frame, text="Filter Column:").grid(row=4, column=0, padx=5)
        self.filter_col_var = tk.StringVar()
        self.filter_col_dropdown = ttk.Combobox(self.control_frame, textvariable=self.filter_col_var, state="readonly")
        self.filter_col_dropdown.grid(row=4, column=1, padx=5)
        
        ttk.Label(self.control_frame, text="Filter Value:").grid(row=5, column=0, padx=5)
        self.filter_val_var = tk.StringVar()
        self.filter_val_entry = ttk.Entry(self.control_frame, textvariable=self.filter_val_var, state="disabled")
        self.filter_val_entry.grid(row=5, column=1, padx=5)
        
        # Enable filter value entry when column is selected
        self.filter_col_var.trace_add('write', lambda *_: self.filter_val_entry.config(
            state="normal" if self.filter_col_var.get() else "disabled"
        ))
        
        # Action buttons
        self.load_btn = ttk.Button(self.nav_frame, text="Load Data", command=self.load_data)
        self.load_btn.grid(row=0, column=0, pady=5, sticky="ew")
        
        self.plot_btn = ttk.Button(self.nav_frame, text="Generate Plot", command=self.generate_plot)
        self.plot_btn.grid(row=6, column=0, pady=5, sticky="ew")
        
        self.fit_btn = ttk.Button(self.nav_frame, text="Apply Fit", command=self.apply_fit)
        self.fit_btn.grid(row=7, column=0, pady=5, sticky="ew")
        
        self.reset_btn = ttk.Button(self.nav_frame, text="Reset Plot", command=self.reset_plot)
        self.reset_btn.grid(row=8, column=0, pady=5, sticky="ew")
        
        # Export Model button (initially disabled)
        self.export_btn = ttk.Button(
            self.nav_frame, 
            text="Export Model", 
            command=self.export_model,
            state=tk.DISABLED
        )
        self.export_btn.grid(row=9, column=0, pady=5, sticky="ew")
        
        # Units confirmation button
        self.units_btn = ttk.Button(self.nav_frame, text="Confirm Units", command=self._update_units)
        self.units_btn.grid(row=10, column=0, pady=5, sticky="ew")
        
        # Results display frame
        self.results_frame = ttk.LabelFrame(self.nav_frame, text="Fit Results", padding=10)
        self.results_frame.grid(row=11, column=0, pady=5, sticky="nsew")
        
        # Results text with scrollbars
        self.results_text = tk.Text(self.results_frame, height=15, width=50, wrap=tk.NONE)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        self.results_yscroll = ttk.Scrollbar(self.results_frame, command=self.results_text.yview)
        self.results_yscroll.grid(row=0, column=1, sticky="ns")
        
        self.results_xscroll = ttk.Scrollbar(self.results_frame, orient="horizontal", command=self.results_text.xview)
        self.results_xscroll.grid(row=1, column=0, sticky="ew")
        
        self.results_text.config(
            yscrollcommand=self.results_yscroll.set,
            xscrollcommand=self.results_xscroll.set
        )
        
        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.nav_frame.rowconfigure(11, weight=1)
        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
    
    def _update_units(self):
        """Update the stored units when user confirms"""
        self.x_units = self.x_units_var.get()
        self.y_units = self.y_units_var.get()
        
        # Refresh plot if we have data
        if self.data is not None:
            self.generate_plot()
    
    def _get_axis_title(self, col_name, units):
        """Generate axis title with optional units"""
        title = col_name
        if units:
            title += f"\n({units})"
        return title
    
    def reset_plot(self):
        """Reset the plot to show only the scatterplot of original data"""
        if self.data is None:
            self._show_error("No data loaded")
            return
            
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        
        if not x_col or not y_col:
            self._show_error("Please select both x and y columns")
            return
        
        filtered_data = self._apply_filter()
        
        self.ax.clear()
        self.ax.scatter(filtered_data[x_col], filtered_data[y_col], label='Data')
        self.ax.set_xlabel(self._get_axis_title(x_col, self.x_units))
        self.ax.set_ylabel(self._get_axis_title(y_col, self.y_units))
        title = f"{y_col} vs {x_col}"
        if self.filter_col_var.get():
            title += f" (Filter: {self.filter_col_var.get()} = {self.filter_val_var.get()})"
        self.ax.set_title(title)
        self.ax.legend()
        
        # Clear any stored fit
        self.current_fit = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Disable export button when plot is reset
        self.export_btn.config(state=tk.DISABLED)
        
        self.canvas.draw()
    
    def load_data(self):
        """Load data from file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if not filepath:
            return
            
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            else:
                self.data = pd.read_excel(filepath)
                
            self.update_display()
            self.update_dropdowns()
        except Exception as e:
            self._show_error(f"Error loading file: {str(e)}")
    
    def update_display(self):
        """Update the data display"""
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, self.data.head().to_string())
        self.data_text.config(state=tk.DISABLED)
    
    def update_dropdowns(self):
        """Update the dropdown menus with current columns"""
        columns = self.data.columns.tolist()
        self.x_dropdown['values'] = columns
        self.y_dropdown['values'] = columns
        self.filter_col_dropdown['values'] = columns
        
        if len(columns) > 0:
            self.x_var.set(columns[0])
            self.y_var.set(columns[1] if len(columns) > 1 else columns[0])
            self.filter_col_var.set('')  # Start with no filter selected
    
    def _apply_filter(self):
        """Filter the data based on selected column and value"""
        if not self.filter_col_var.get() or not self.filter_val_var.get():
            return self.data  # Return all data if no filter selected
        
        try:
            filter_col = self.filter_col_var.get()
            filter_val = self.filter_val_var.get()
            
            # Try to convert to numeric if possible
            try:
                filter_val = float(filter_val)
                return self.data[self.data[filter_col] == filter_val]
            except ValueError:
                # Handle string comparison if conversion fails
                return self.data[self.data[filter_col].astype(str) == str(filter_val)]
        except Exception as e:
            self._show_error(f"Filter error: {str(e)}")
            return self.data
    
    def generate_plot(self):
        """Generate the initial scatter plot"""
        if self.data is None:
            self._show_error("No data loaded")
            return
            
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        
        if not x_col or not y_col:
            self._show_error("Please select both x and y columns")
            return
        
        filtered_data = self._apply_filter()
        
        self.ax.clear()
        self.ax.scatter(filtered_data[x_col], filtered_data[y_col], label='Data')
        self.ax.set_xlabel(self._get_axis_title(x_col, self.x_units))
        self.ax.set_ylabel(self._get_axis_title(y_col, self.y_units))
        title = f"{y_col} vs {x_col}"
        if self.filter_col_var.get():
            title += f" (Filter: {self.filter_col_var.get()} = {self.filter_val_var.get()})"
        self.ax.set_title(title)
        
        if self.current_fit is not None:
            self._plot_fit_curve(filtered_data)
        
        self.ax.legend()
        self.canvas.draw()
    
    def apply_fit(self):
        """Apply the selected fit model after resetting the plot"""
        if self.data is None:
            self._show_error("No data loaded")
            return
            
        try:
            self.alpha = float(self.alpha_var.get())
            if not (0 < self.alpha < 1):
                raise ValueError("Alpha must be between 0 and 1")
        except ValueError as e:
            self._show_error(f"Invalid confidence level: {str(e)}")
            return
            
        x_col = self.x_var.get()
        y_col = self.y_var.get()
        model_name = self.model_var.get()
        
        if not all([x_col, y_col, model_name]):
            self._show_error("Please select columns and model")
            return
        
        # First reset the plot (similar to reset_plot functionality)
        filtered_data = self._apply_filter()
        
        self.ax.clear()
        self.ax.scatter(filtered_data[x_col], filtered_data[y_col], label='Data')
        self.ax.set_xlabel(self._get_axis_title(x_col, self.x_units))
        self.ax.set_ylabel(self._get_axis_title(y_col, self.y_units))
        title = f"{y_col} vs {x_col}"
        if self.filter_col_var.get():
            title += f" (Filter: {self.filter_col_var.get()} = {self.filter_val_var.get()})"
        self.ax.set_title(title)
        
        # Clear any stored fit
        self.current_fit = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        self.canvas.draw()
        
        # Now proceed with fitting
        try:
            # Initialize and fit the model
            model_class = self.MODELS[model_name]
            model = model_class()
            model.fit(
                filtered_data[x_col].to_numpy(),
                filtered_data[y_col].to_numpy()
            )
            
            # Store results
            self.current_fit = {
                'model': model,
                'name': model_name,
                'x_col': x_col,
                'y_col': y_col,
                'filtered_data': filtered_data
            }
            
            self._plot_fit_curve(filtered_data)
            self._display_fit_details(model)
            
            # Enable export button after successful fit
            self.export_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self._show_error(f"Fit failed: {str(e)}")
    
    def export_model(self):
        """Export the current model with plot and details"""
        if not self.current_fit:
            self._show_error("No model to export")
            return
        
        # Step 1: Select parent directory
        parent_dir = filedialog.askdirectory(title="Select Parent Directory")
        if not parent_dir:
            return  # User cancelled
        
        # Step 2: Get new directory name
        dir_name = self._get_user_input("Enter New Directory Name", "Enter name for new directory:")
        if not dir_name:
            return  # User cancelled
        
        # Create the new directory
        export_dir = os.path.join(parent_dir, dir_name)
        try:
            os.makedirs(export_dir, exist_ok=True)
        except Exception as e:
            self._show_error(f"Error creating directory: {str(e)}")
            return
        
        # Step 3: Get model name
        model_name = self._get_user_input("Enter Model Name", "Enter name for the model:")
        if not model_name:
            return  # User cancelled
        
        # Save plot
        plot_path = os.path.join(export_dir, f"{model_name}-plot.png")
        try:
            self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            self._show_error(f"Error saving plot: {str(e)}")
            return
        
        # Save model
        model_path = os.path.join(export_dir, f"{model_name}_model.obj")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.current_fit['model'], f)
        except Exception as e:
            self._show_error(f"Error saving model: {str(e)}")
            return
        
        # Save details
        details_path = os.path.join(export_dir, f"{model_name}_details.txt")
        try:
            with open(details_path, 'w') as f:
                f.write(self._generate_details_string(self.current_fit['model']))
        except Exception as e:
            self._show_error(f"Error saving details: {str(e)}")
            return
        
        # Show success message
        messagebox.showinfo(
            "Export Successful",
            f"Model exported to:\n{export_dir}\n\n"
            f"Files created:\n"
            f"- {model_name}-plot.png\n"
            f"- {model_name}_model.obj\n"
            f"- {model_name}_details.txt"
        )
    
    def _get_user_input(self, title, prompt):
        """Helper to get user input through a dialog"""
        dialog = tk.Toplevel(self.master)
        dialog.title(title)
        
        ttk.Label(dialog, text=prompt).pack(padx=10, pady=5)
        
        entry_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=entry_var)
        entry.pack(padx=10, pady=5)
        
        result = None
        
        def on_confirm():
            nonlocal result
            result = entry_var.get()
            dialog.destroy()
        
        def on_cancel():
            nonlocal result
            result = None
            dialog.destroy()
        
        ttk.Button(dialog, text="Confirm", command=on_confirm).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(dialog, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=10, pady=10)
        
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)
        
        return result
    
    def _plot_fit_curve(self, filtered_data):
        """Plot the fit curve and confidence intervals"""
        if not self.current_fit:
            return
            
        model = self.current_fit['model']
        x_col = self.current_fit['x_col']
        y_col = self.current_fit['y_col']
        
        # Generate fit curve
        xx = np.linspace(
            min(filtered_data[x_col]), 
            max(filtered_data[x_col]), 
            100
        )
        yy = model.predict(xx)
        
        # Plot data and fit
        self.ax.scatter(filtered_data[x_col], filtered_data[y_col], label='Data')
        self.ax.plot(xx, yy, 'r-', label=f"{self.current_fit['name']} Fit")
        
        # Plot confidence intervals
        if hasattr(model, 'upper_pred_edge'):
            if self.current_fit['name'] in ['LLSR', 'PLSR']:
                # Least Squares - use alpha parameter
                upper = model.upper_pred_edge(xx, alpha=self.alpha)
                lower = model.lower_pred_edge(xx, alpha=self.alpha)
                label = f'{100*(1-self.alpha):.0f}% CI'
            else:
                # Quantile Regression - no alpha parameter
                upper = model.upper_pred_edge(xx)
                lower = model.lower_pred_edge(xx)
                label = 'Prediction Band'
            
            self.ax.plot(xx, upper, 'g--', label=f'Upper {label}')
            self.ax.plot(xx, lower, 'g--', label=f'Lower {label}')
        
        self.ax.set_xlabel(self._get_axis_title(x_col, self.x_units))
        self.ax.set_ylabel(self._get_axis_title(y_col, self.y_units))
        title = f"{y_col} vs {x_col} with {self.current_fit['name']} Fit"
        if self.filter_col_var.get():
            title += f" (Filter: {self.filter_col_var.get()} = {self.filter_val_var.get()})"
        self.ax.set_title(title)
        self.ax.legend()
        self.canvas.draw()
    
    def _display_fit_details(self, model):
        """Display detailed fit results"""
        details_str = self._generate_details_string(model)
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, details_str)
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)
    
    def _generate_details_string(self, model):
        """Generate the detailed fit report string"""
        details_str = f"{self.current_fit['name']} Fit Details\n"
        details_str += "="*60 + "\n"
        
        # Model-specific details
        if self.current_fit['name'] == 'LLSR':
            details_str += 'Functional Form: f(x) = p0 + p1*x\n'
            details_str += f'[p0, p1] = {model.fit_params}\n'
        elif self.current_fit['name'] == 'PLSR':
            details_str += 'Functional Form: f(x) = p0 + p1*x + p2*x²\n'
            details_str += f'[p0, p1, p2] = {model.fit_params}\n'
        elif self.current_fit['name'] == 'LQR':
            details_str += 'Functional Form: f(x) = p0 + p1*x\n'
            details_str += f'[p0, p1] = {model.fit_params}\n'
        elif self.current_fit['name'] == 'PQR':
            details_str += 'Functional Form: f(x) = p0 + p1*x + p2*x²\n'
            details_str += f'[p0, p1, p2] = {model.fit_params}\n'
        elif self.current_fit['name'] == 'PowQR':
            details_str += 'Functional Form: f(x) = p0 + p1*x^b\n'
            details_str += f'Parameters = {model.fit_params}\n'
        
        # Common model attributes
        details_str += "\nCovariance Matrix:\n"
        details_str += str(model.covar) + "\n" if hasattr(model, 'covar') else "Not available\n"
        
        details_str += "="*60 + "\n"
        details_str += f"Confidence Level: {100*(1-self.alpha):.1f}%\n"
        details_str += f"s (std dev): {model.s}\n" if hasattr(model, 's') else ""
        details_str += f"Degrees of freedom: {model.df}\n" if hasattr(model, 'df') else ""
        details_str += f"Elow: {model.Elow}\n" if hasattr(model, 'Elow') else ""
        details_str += f"Eupp: {model.Eupp}\n" if hasattr(model, 'Eupp') else ""
        details_str += f"imin_est: {model.imin_est}\n" if hasattr(model, 'imin_est') else ""
        details_str += f"imax_est: {model.imax_est}\n" if hasattr(model, 'imax_est') else ""
        details_str += f"conc_est_min: {model.conc_est_min}\n" if hasattr(model, 'conc_est_min') else ""
        details_str += f"conc_est_max: {model.conc_est_max}\n" if hasattr(model, 'conc_est_max') else ""
        
        # Additional model-specific attributes
        if hasattr(model, 'smoothing'):
            details_str += f"Smoothing: {model.smoothing}\n"
        if hasattr(model, 'baseline'):
            details_str += f"Baseline: {model.baseline}\n"
        if hasattr(model, 'calibration'):
            details_str += f"Calibration: {model.calibration}\n"
        
        # Add filter info if applied
        if self.filter_col_var.get():
            details_str += "\n" + "="*60 + "\n"
            details_str += f"Filter Applied: {self.filter_col_var.get()} = {self.filter_val_var.get()}\n"
            details_str += f"Filtered Data Points: {len(self.current_fit['filtered_data'])}"
        
        return details_str
    
    def _show_error(self, message):
        """Display error messages"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}")
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)

def main():
    root = tk.Tk()
    app = DataFrameVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()