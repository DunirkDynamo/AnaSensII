import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import least_squares
import scipy.stats

def get_estimate(measured_value, curve_object, curve_errprop, covar_params, df, s, solver_lims):
    def predint_upper(x):
        return curve_object.predict(x) + scipy.stats.t.ppf(0.975, df) * np.sqrt(curve_errprop(x, covar_params) + s**2) - measured_value
    def predint_lower(x):
        return curve_object.predict(x) - scipy.stats.t.ppf(0.975, df) * np.sqrt(curve_errprop(x, covar_params) + s**2) - measured_value
    def best_model(x):
        return curve_object.predict(x) - measured_value

    initial_guess = np.mean(solver_lims)
    upper_conc = least_squares(predint_lower, x0=initial_guess, bounds=solver_lims).x[0]
    best_conc = least_squares(best_model, x0=initial_guess, bounds=solver_lims).x[0]
    lower_conc = least_squares(predint_upper, x0=initial_guess, bounds=solver_lims).x[0]
    
    return np.array([lower_conc, best_conc, upper_conc])

def get_QR_estimate(measured_value, curve_object, solver_lims):
    def predint_upper(x):
        return curve_object.upper_pred_edge(x) - measured_value
    def predint_lower(x):
        return curve_object.lower_pred_edge(x) - measured_value
    def best_model(x):
        return curve_object.predict(x) - measured_value

    initial_guess = np.mean(solver_lims)
    upper_conc = least_squares(predint_lower, x0=initial_guess, bounds=solver_lims).x[0]
    best_conc = least_squares(best_model, x0=initial_guess, bounds=solver_lims).x[0]
    lower_conc = least_squares(predint_upper, x0=initial_guess, bounds=solver_lims).x[0]
    
    return np.array([lower_conc, best_conc, upper_conc])

class VoltammogramAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Voltammogram Analyzer")
        self.voltammogram_list = []
        self.std_curve = None
        self.regression_type = None
        self.calibrated_metric = None  # Properly initialized
        self.current_plot_index = 0
        self.generated_plots = []
        
        self.load_voltammograms()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def load_voltammograms(self):
        self.clear_window()
        tk.Label(self.root, text="Select Folder Containing Voltammograms", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Button(self.root, text="Browse Folder", 
                 command=self.browse_voltammograms).pack(pady=20)

    def browse_voltammograms(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Voltammograms")
        if not folder_path:
            return
            
        try:
            self.voltammogram_list = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.pkl', '.pickle')):
                    with open(os.path.join(folder_path, filename), 'rb') as f:
                        vgram = pickle.load(f)
                        if isinstance(vgram, dict) and 'calibration_table' in vgram:
                            vgram['filepath'] = os.path.join(folder_path, filename)
                            self.voltammogram_list.append(vgram)
            
            if not self.voltammogram_list:
                messagebox.showerror("Error", "No valid voltammograms found!")
                return
                
            # Get columns from first voltammogram's calibration table
            cols = self.voltammogram_list[0]['calibration_table'].columns.tolist()
            self.prompt_calibration_metric(cols)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load voltammograms: {str(e)}")

    def prompt_calibration_metric(self, columns):
        self.clear_window()
        tk.Label(self.root, text="Select Calibration Metric Column", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.metric_var = tk.StringVar()
        metric_dropdown = ttk.Combobox(self.root, textvariable=self.metric_var, 
                                      values=columns, state='readonly', width=30)
        metric_dropdown.pack(pady=10)
        metric_dropdown.current(0)
        
        tk.Button(self.root, text="Confirm Selection", 
                 command=self.set_calibration_metric).pack(pady=20)

    def set_calibration_metric(self):
        selected_metric = self.metric_var.get()
        if not selected_metric:
            messagebox.showwarning("Warning", "Please select a metric column")
            return
        self.calibrated_metric = selected_metric
        self.load_standard_curve()

    def load_standard_curve(self):
        self.clear_window()
        tk.Label(self.root, text="Select Standard Curve File", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Button(self.root, text="Browse File", 
                 command=self.browse_standard_curve).pack(pady=20)

    def browse_standard_curve(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'rb') as f:
                self.std_curve = pickle.load(f)
            self.show_regression_selection()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load standard curve: {str(e)}")

    def show_regression_selection(self):
        self.clear_window()
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Regression selection
        reg_frame = tk.LabelFrame(main_frame, text="Regression Type Selection",
                                font=('Arial', 10, 'bold'))
        reg_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(reg_frame, text="Select Regression Method:").pack(side=tk.LEFT, padx=5)
        
        self.reg_var = tk.StringVar()
        reg_options = ['Quantile Regression', 'Least Squares Regression']
        reg_dropdown = ttk.Combobox(reg_frame, textvariable=self.reg_var, 
                                   values=reg_options, state='readonly', width=25)
        reg_dropdown.pack(side=tk.LEFT, padx=5)
        reg_dropdown.current(0)
        
        confirm_btn = tk.Button(reg_frame, text="Confirm", 
                              command=self.set_regression_type)
        confirm_btn.pack(side=tk.LEFT, padx=10)
        
        # Continue button
        self.continue_btn = tk.Button(main_frame, text="Start Analysis", 
                                    state=tk.DISABLED, command=self.perform_analysis)
        self.continue_btn.pack(pady=20)

    def set_regression_type(self):
        selection = self.reg_var.get()
        if not selection:
            messagebox.showwarning("Warning", "Please select a regression type")
            return
            
        self.regression_type = "quantile" if "Quantile" in selection else "least_squares"
        self.continue_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Selection Confirmed", f"Using {selection}")

    def perform_analysis(self):
        if not self.calibrated_metric:
            messagebox.showerror("Error", "No calibration metric selected!")
            return
            
        conc_limits = np.array([self.std_curve.conc_std_min, self.std_curve.conc_std_max])
        
        for voltammogram in self.voltammogram_list:
            ct = voltammogram['calibration_table']
            if self.calibrated_metric not in ct.columns:
                continue
                
            measurements = ct[self.calibrated_metric].values
            estimates = []
            
            for mv in measurements:
                try:
                    if self.regression_type == "quantile":
                        low, est, upp = get_QR_estimate(mv, self.std_curve, conc_limits)
                    else:
                        low, est, upp = get_estimate(
                            mv,
                            self.std_curve,
                            self.std_curve.errprop,
                            self.std_curve.covar,
                            self.std_curve.df,
                            self.std_curve.s,
                            conc_limits
                        )
                    estimates.append({
                        'measured': mv,
                        'estimate': est,
                        'low_limit': low,
                        'upp_limit': upp
                    })
                except Exception as e:
                    estimates.append({
                        'measured': mv,
                        'error': str(e)
                    })
            
            voltammogram['estimates'] = estimates
        
        success_count = sum(1 for v in self.voltammogram_list if 'estimates' in v)
        messagebox.showinfo("Analysis Complete", 
                          f"Processed {success_count}/{len(self.voltammogram_list)} voltammograms")
        self.show_results()

    def show_results(self):
        self.clear_window()
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(main_frame, text="Analysis Results", 
                                    font=('Arial', 10, 'bold'))
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                 font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Populate results
        self.results_text.insert(tk.END, f"{'Name':<20}{'Measured':>12}{'Estimate':>12}{'Low Limit':>12}{'Upper Limit':>12}\n")
        self.results_text.insert(tk.END, "-"*68 + "\n")
        
        for vgram in self.voltammogram_list:
            name = vgram.get('name', 'Unnamed')
            estimates = vgram.get('estimates', [])
            for est in estimates:
                if 'error' in est:
                    self.results_text.insert(tk.END, 
                        f"{name:<20} {est['measured']:>12.3f} {'ERROR':>12} {est['error']}\n")
                else:
                    self.results_text.insert(tk.END, 
                        f"{name:<20} {est['measured']:>12.3f} {est['estimate']:>12.3f} "
                        f"{est['low_limit']:>12.3f} {est['upp_limit']:>12.3f}\n")
        
        self.results_text.config(state=tk.DISABLED)
        
        # Add export button
        tk.Button(main_frame, text="Export Results to CSV", 
                 command=self.export_results).pack(pady=10)

    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Measured Value', 'Estimate', 'Low Limit', 'Upper Limit', 'Error'])
                
                for vgram in self.voltammogram_list:
                    name = vgram.get('name', 'Unnamed')
                    for est in vgram.get('estimates', []):
                        writer.writerow([
                            name,
                            est.get('measured', ''),
                            est.get('estimate', ''),
                            est.get('low_limit', ''),
                            est.get('upp_limit', ''),
                            est.get('error', '')
                        ])
            messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Error exporting results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = VoltammogramAnalyzer(root)
    root.mainloop()