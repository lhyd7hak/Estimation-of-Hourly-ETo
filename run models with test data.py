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
        self.root.geometry("600x450")
        
        # File paths
        self.csv_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.scaler_x_path = tk.StringVar()
        self.scaler_y_path = tk.StringVar()
        
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

        # --- Info Section ---
        info_frame = ttk.LabelFrame(self.root, text="2. Information", padding="15")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_text = (
            "Model inputs will be automatically detected from the .keras filename.\n"
            "Data scaling and column ordering are handled automatically based on the Scaler file."
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

        target_var = 'PM ETo (mm)'

        try:
            # 2. Auto-Detect Inputs from Filename
            selected_inputs = self.extract_inputs_from_filename(self.model_path.get())
            print(f"Auto-detected inputs: {selected_inputs}")

           # 3. Load Data & Scalers
            df_test = pd.read_csv(self.csv_path.get())
            
            # --- THE FIX: Clean the data by dropping rows with missing values ---
            # We only check the specific inputs we need, plus the target variable
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
            # --------------------------------------------------------------------

            scaler_x = joblib.load(self.scaler_x_path.get())
            scaler_y = joblib.load(self.scaler_y_path.get())
            
            # ---> THE MISSING DEFINITION IS RESTORED HERE <---
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

            # 4. Transform Data
            # Transform all columns the scaler expects, exactly in the order it expects them
            X_all_scaled = scaler_x.transform(df_test[all_scaler_cols])
            
            # Find the indices of the selected variables to slice the array mathematically
            required_indices = [all_scaler_cols.index(col) for col in selected_inputs]
            X_test_scaled = X_all_scaled[:, required_indices]
            
            y_test = df_test[target_var].values

            # 4. Transform Data
            # Transform all columns the scaler expects, exactly in the order it expects them
            X_all_scaled = scaler_x.transform(df_test[all_scaler_cols])
            
            # Find the indices of the selected variables to slice the array mathematically
            required_indices = [all_scaler_cols.index(col) for col in selected_inputs]
            X_test_scaled = X_all_scaled[:, required_indices]
            
            y_test = df_test[target_var].values

            # 5. Load Model & Predict
            print("Loading Keras model...")
            model = tf.keras.models.load_model(self.model_path.get())
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            # 6. Calculate Metrics
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

            # 7. Save Results & Plot
            model_dir = Path(self.model_path.get()).parent
            model_name = Path(self.model_path.get()).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save CSV
            results_df = pd.DataFrame({"RMSE": [rmse], "R2": [r2], "MBE": [mbe], "MSE": [mse], "NSE": [nse], "d": [d]})
            csv_save_path = model_dir / f"{model_name}_TestResults_{timestamp}.csv"
            results_df.to_csv(csv_save_path, index=False)

            # Generate Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            hexbin = ax.hexbin(y_test, y_pred, gridsize=30, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Measured ETo (mm)', fontsize=12)
            ax.set_ylabel('Predicted ETo (mm)', fontsize=12)
            ax.set_title(f'Test Results: {model_name}\nRMSE = {rmse:.4f} | R² = {r2:.4f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            plt.colorbar(hexbin, ax=ax, label='Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save Plot
            plot_save_path = model_dir / f"{model_name}_Hexbin_{timestamp}.png"
            plt.savefig(plot_save_path, dpi=300)
            
            messagebox.showinfo("Success", f"Inference complete!\n\nDetected Inputs: {', '.join(selected_inputs)}\n\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n\nResults and plot saved to model folder.")
            plt.show()

        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EToAutoInferenceApp(root)
    root.mainloop()