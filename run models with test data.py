import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

class EToAutoInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ETo Neural Network Auto-Inference Tool")
        self.root.geometry("650x500")  # Slightly larger to accommodate new selector
        
        # File paths
        self.csv_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.scaler_x_path = tk.StringVar()
        self.scaler_y_path = tk.StringVar()
        self.output_dir = tk.StringVar()  # New variable for output directory
        
        # Dictionary to map your filename abbreviations back to the exact CSV columns
        self.column_map = {
            'AirTempC': 'Air Temp (C)',
            'SolRadWsq.m': 'Sol Rad (W/sq.m)',
            'RelHum': 'Rel Hum (%)',
            'WindSpeedms': 'Wind Speed (m/s)'
        }
        
        self.create_widgets()

    def create_widgets(self):
        # --- File Selection Section ---
        file_frame = ttk.LabelFrame(self.root, text="1. Select Files", padding="15")
        file_frame.pack(fill="x", padx=10, pady=10)

        self.add_file_selector(file_frame, "Test Data (CSV):", self.csv_path, 0, [("CSV Files", "*.csv")])
        self.add_file_selector(file_frame, "Trained Model (.keras):", self.model_path, 1, [("Keras Models", "*.keras")])
        self.add_file_selector(file_frame, "Scaler X (.pkl):", self.scaler_x_path, 2, [("Pickle Files", "*.pkl")])
        self.add_file_selector(file_frame, "Scaler Y (.pkl):", self.scaler_y_path, 3, [("Pickle Files", "*.pkl")])
        
        # --- Output Directory Selection Section ---
        output_frame = ttk.LabelFrame(self.root, text="2. Select Output Folder", padding="15")
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(output_frame, text="Results Folder:").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=10, pady=8)
        ttk.Button(output_frame, text="Browse", command=self.select_output_folder).grid(row=0, column=2, pady=8)
        
        # Optional: Add option to use default folder (results folder next to model)
        self.use_default = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Use default 'results' folder next to model", 
                       variable=self.use_default, command=self.toggle_output_folder).grid(row=1, column=0, columnspan=3, pady=5)

        # --- Info Section ---
        info_frame = ttk.LabelFrame(self.root, text="3. Information", padding="15")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_text = (
            "Model inputs will be automatically detected from the .keras filename.\n"
            "Data scaling and column ordering are handled automatically based on the Scaler file.\n"
            "Results will be saved in your chosen output folder."
        )
        ttk.Label(info_frame, text=info_text, justify="center").pack()

        # --- Run Section ---
        run_frame = ttk.Frame(self.root, padding="10")
        run_frame.pack(fill="x", padx=10, pady=10)

        self.run_btn = ttk.Button(run_frame, text="Run Inference & Generate Plot", command=self.run_inference)
        self.run_btn.pack(fill="x", ipady=8)

    def add_file_selector(self, parent, label_text, string_var, row, filetypes):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=string_var, width=50).grid(row=row, column=1, padx=10, pady=8)
        ttk.Button(parent, text="Browse", command=lambda: string_var.set(filedialog.askopenfilename(filetypes=filetypes))).grid(row=row, column=2, pady=8)
    
    def select_output_folder(self):
        """Open dialog to select output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder for Results")
        if folder:
            self.output_dir.set(folder)
            self.use_default.set(False)  # Uncheck default option when user selects folder
    
    def toggle_output_folder(self):
        """Enable/disable output folder entry based on checkbox"""
        if self.use_default.get():
            self.output_dir.set("")  # Clear custom folder
            # Disable the entry and browse button
            # You could implement this if desired

    def extract_inputs_from_filename(self, model_filepath):
        """Extracts the expected input columns from the model's filename."""
        filename = Path(model_filepath).stem  # Gets the name without .keras
        
        # Split at '_Arch' to isolate the input variables section
        try:
            inputs_string = filename.split('_Arch')[0]
            input_tokens = inputs_string.split('_')
            
            selected_inputs = []
            for token in input_tokens:
                if token in self.column_map:
                    selected_inputs.append(self.column_map[token])
                else:
                    raise ValueError(f"Unknown variable abbreviation '{token}' in filename.")
            
            return selected_inputs
        except Exception as e:
            raise ValueError(f"Could not parse filename '{filename}'. Ensure it follows the format 'Var1_Var2_Arch...': {e}")

    def run_inference(self):
        # 1. Validate Selections
        if not all([self.csv_path.get(), self.model_path.get(), self.scaler_x_path.get(), self.scaler_y_path.get()]):
            messagebox.showerror("Error", "Please select all four required files.")
            return
        
        # 2. Determine output directory
        if self.use_default.get():
            # Use default 'results' folder next to model
            model_dir = Path(self.model_path.get()).parent
            output_dir = model_dir / 'results'
        elif self.output_dir.get():
            # Use user-selected folder
            output_dir = Path(self.output_dir.get())
        else:
            # No output folder selected, ask user
            messagebox.showerror("Error", "Please select an output folder or check 'Use default results folder'.")
            return
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        print(f"\nResults will be saved to: {output_dir}")

        target_var = 'PM ETo (mm)'

        try:
            # 3. Auto-Detect Inputs from Filename
            selected_inputs = self.extract_inputs_from_filename(self.model_path.get())
            print(f"Auto-detected inputs: {selected_inputs}")

            # 4. Load Data & Scalers
            df_test = pd.read_csv(self.csv_path.get())
            
            # Print column names for debugging
            print("\nAvailable columns in CSV:")
            print(df_test.columns.tolist())
            
            # Clean the data by dropping rows with missing values
            cols_to_check = selected_inputs + [target_var]
            
            # Count original rows
            original_len = len(df_test)
            
            # Drop any row that has a blank cell in our required columns
            df_test = df_test.dropna(subset=cols_to_check)
            
            # See how many were dropped
            cleaned_len = len(df_test)
            if cleaned_len < original_len:
                print(f"Data Cleanup: Dropped {original_len - cleaned_len} rows containing blank (NaN) data.")
                
            if cleaned_len == 0:
                raise ValueError("After removing blank rows, there is no data left to process! Please check your CSV.")

            scaler_x = joblib.load(self.scaler_x_path.get())
            scaler_y = joblib.load(self.scaler_y_path.get())
            
            # Extract expected columns directly from the scaler to prevent order mismatch
            try:
                all_scaler_cols = list(scaler_x.feature_names_in_)
            except AttributeError:
                messagebox.showerror("Scaler Error", "Scaler is missing feature names. Ensure it was fitted on a DataFrame.")
                return

            # Verify selected inputs exist in the scaler and the CSV
            for col in selected_inputs:
                if col not in all_scaler_cols:
                    raise ValueError(f"Auto-detected input '{col}' was not found in the loaded Scaler.")
                if col not in df_test.columns:
                    raise ValueError(f"Auto-detected input '{col}' was not found in the CSV file.")

            # 5. Transform Data
            # Transform all columns the scaler expects, exactly in the order it expects them
            X_all_scaled = scaler_x.transform(df_test[all_scaler_cols])
            
            # Find the indices of the selected variables to slice the array mathematically
            required_indices = [all_scaler_cols.index(col) for col in selected_inputs]
            X_test_scaled = X_all_scaled[:, required_indices]
            
            y_test = df_test[target_var].values

            # 6. Load Model & Predict
            print("Loading Keras model...")
            model = tf.keras.models.load_model(self.model_path.get())
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            # 7. Calculate Metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mbe = np.mean(y_pred - y_test)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            nse = 1 - (ss_res / ss_tot)
            
            d_numerator = np.sum((y_pred - y_test) ** 2)
            d_denominator = np.sum((np.abs(y_pred - np.mean(y_test)) + np.abs(y_test - np.mean(y_test))) ** 2)
            d = 1 - (d_numerator / d_denominator)

            # 8. Save Results
            model_name = Path(self.model_path.get()).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create detailed results CSV with the correct column names from your file
            results_data = {
                'Stn_Id': df_test['Stn Id'],
                'Stn_Name': df_test['Stn Name'],
                'Date': df_test['Date'],
                'Hour': df_test['Hour (PST)'],
                'ETo_Measured_mm': y_test,
                'ETo_Predicted_mm': y_pred,
                'Error_mm': y_pred - y_test
            }
            
            # Create DataFrame with detailed results
            detailed_results_df = pd.DataFrame(results_data)
            
            # Save detailed results CSV
            detailed_csv_path = output_dir / f"{model_name}_DetailedResults_{timestamp}.csv"
            detailed_results_df.to_csv(detailed_csv_path, index=False)
            print(f"Detailed results saved to: {detailed_csv_path}")
            
            # Save metrics summary CSV
            metrics_df = pd.DataFrame({
                "RMSE_mm": [rmse], 
                "R2": [r2], 
                "MBE_mm": [mbe], 
                "MSE_mm2": [mse], 
                "NSE": [nse], 
                "d": [d],
                "Number_of_Samples": [len(y_test)]
            })
            metrics_csv_path = output_dir / f"{model_name}_MetricsSummary_{timestamp}.csv"
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Metrics summary saved to: {metrics_csv_path}")

            # Generate Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            hexbin = ax.hexbin(y_test, y_pred, gridsize=30, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Measured ETo (mm)', fontsize=12)
            ax.set_ylabel('Predicted ETo (mm)', fontsize=12)
            ax.set_title(f'Test Results: {model_name}\nRMSE = {rmse:.4f} mm | R² = {r2:.4f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            plt.colorbar(hexbin, ax=ax, label='Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save Plot
            plot_save_path = output_dir / f"{model_name}_Hexbin_{timestamp}.png"
            plt.savefig(plot_save_path, dpi=300)
            print(f"Plot saved to: {plot_save_path}")
            
            # Display first few rows of detailed results in console
            print("\nFirst 5 rows of detailed results:")
            print(detailed_results_df.head())
            
            # Display summary statistics
            print("\nSummary Statistics:")
            print(f"Mean Error: {np.mean(y_pred - y_test):.4f} mm")
            print(f"Std Dev Error: {np.std(y_pred - y_test):.4f} mm")
            
            messagebox.showinfo("Success", 
                f"Inference complete!\n\n"
                f"Detected Inputs: {', '.join(selected_inputs)}\n\n"
                f"RMSE: {rmse:.4f} mm\n"
                f"R²: {r2:.4f}\n"
                f"Samples: {len(y_test)}\n\n"
                f"Results saved to:\n{output_dir}\n\n"
                f"Files created:\n"
                f"• {detailed_csv_path.name}\n"
                f"• {metrics_csv_path.name}\n"
                f"• {plot_save_path.name}")
            plt.show()

        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = EToAutoInferenceApp(root)
    root.mainloop()
