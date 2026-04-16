import pandas as pd
import numpy as np
import os
import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# =================================================================
# 1. USER CONTROL PANEL
# =================================================================
BASE_PATH = r"D:\SSD Allen CoAgMet hourly"

# Locations of train and test files.
TRAIN_FILES = [os.path.join(BASE_PATH, "train.parquet")]

TEST_FILES = [os.path.join(BASE_PATH, "test.parquet")]

TARGET_VAR = 'PM ETo (mm)'

# The 7 Input Sets (Fixed quotes and column names to match CIMIS headers)
INPUT_COMBOS = [
    ['Air Temp (C)', 'Sol Rad (W/sq.m)'],                               # Set 1
    ['Air Temp (C)'],                                                   # Set 2
    ['Air Temp (C)', 'Rel Hum (%)'],                                    # Set 3 
    ['Air Temp (C)', 'Sol Rad (W/sq.m)', 'Rel Hum (%)'],                # Set 4
    ['Air Temp (C)', 'Sol Rad (W/sq.m)', 'Rel Hum (%)', 'Wind Speed (m/s)'], # Set 5
    ['Sol Rad (W/sq.m)', 'Rel Hum (%)'],                                # Set 6
    ['Air Temp (C)', 'Wind Speed (m/s)']                                # Set 7 
]

ARCHITECTURES = [[4], [8, 4], [16, 8, 4], [32, 16, 8, 4]]
ACTIVATIONS = ['relu', 'sigmoid', 'tanh']

# Training Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 2048
EPOCHS = 1000
PATIENCE = 10

# =================================================================
# 2. STATS & UTILS
# =================================================================
def calculate_metrics(obs, pred):
    obs, pred = np.array(obs), np.array(pred)
    mse = np.mean((obs - pred)**2)
    rmse = np.sqrt(mse)
    mbe = np.mean(pred - obs)
    r2 = np.corrcoef(obs, pred, rowvar=False)[0, 1]**2
    
    # Nash-Sutcliffe
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    # Index of Agreement (d)
    denom = np.sum((np.abs(pred - np.mean(obs)) + np.abs(obs - np.mean(obs)))**2)
    d = 1 - (np.sum((obs - pred)**2) / denom)
    
    # Regression Slope & Intercept
    slope, intercept = np.polyfit(obs, pred, 1)
    
    return mse, rmse, mbe, r2, nse, d, slope, intercept

# =================================================================
# 3. GLOBAL DATA PREP (One Scaler Rule)
# =================================================================
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RUN_PATH = os.path.join(BASE_PATH, f"GridRun_{now}")
os.makedirs(RUN_PATH, exist_ok=True)

print("Loading data and fitting Master Scaler...")
df_train = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in TRAIN_FILES])
df_test = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in TEST_FILES])

# Fit Master Scaler on all possible columns
ALL_COLS = list(set([col for sublist in INPUT_COMBOS for col in sublist]))
scaler_X = StandardScaler().fit(df_train[ALL_COLS])
scaler_y = StandardScaler().fit(df_train[[TARGET_VAR]])

# Master Summary File
master_log_path = os.path.join(RUN_PATH, "Master_Summary_Report.csv")
master_results = []

# =================================================================
# 4. GRID SEARCH ENGINE (# MODELS)
# =================================================================
for inp_set in INPUT_COMBOS:
    # Prepare X/y for this input set
    X_train_scaled = scaler_X.transform(df_train[ALL_COLS])
    X_test_scaled = scaler_X.transform(df_test[ALL_COLS])
    
    # Slice only the indices belonging to the current inp_set
    idx = [ALL_COLS.index(c) for c in inp_set]
    X_tr = X_train_scaled[:, idx]
    X_ts = X_test_scaled[:, idx]
    y_tr = scaler_y.transform(df_train[[TARGET_VAR]])

    for layers in ARCHITECTURES:
        for act in ACTIVATIONS:
            model_id = f"{'_'.join(inp_set).replace(' ', '').replace('/', '').replace('(', '').replace(')', '').replace('%', '')}_Arch{'-'.join(map(str, layers))}_{act}"
            print(f"\n>>> Running: {model_id}")
            
            # Subfolder for this model
            model_folder = os.path.join(RUN_PATH, model_id)
            os.makedirs(model_folder, exist_ok=True)

            # Build Model
            model = Sequential()
            model.add(Dense(layers[0], input_dim=len(inp_set), activation=act))
            for nodes in layers[1:]:
                model.add(Dense(nodes, activation=act))
            model.add(Dense(1, activation='linear'))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
            
            # Train
            # Prepare y_test_scaled for validation monitoring
            y_ts_scaled = scaler_y.transform(df_test[[TARGET_VAR]])

            model.fit(X_tr, y_tr, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE,
            # shuffle=False,
              validation_data=(X_ts, y_ts_scaled), 
              callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=0.001)], 
              verbose=0)		     

            # Predict & Unscale
            preds_scaled = model.predict(X_ts, batch_size=BATCH_SIZE)
            y_pred = scaler_y.inverse_transform(preds_scaled).flatten()
            y_true = df_test[TARGET_VAR].values

            # Stats (Unscaled)
            mse, rmse, mbe, r2, nse, d, slope, intercept = calculate_metrics(y_true, y_pred)

            # Save Model Package
            model.save(os.path.join(model_folder, f"{model_id}.keras"))
            joblib.dump(scaler_X, os.path.join(model_folder, f"{model_id}_scalerX.pkl"))
            joblib.dump(scaler_y, os.path.join(model_folder, f"{model_id}_scalerY.pkl"))

            # Individual CSV Report
            df_res = df_test[['Stn Id', 'Date', TARGET_VAR]].copy()
            df_res['Predicted_ETo'] = y_pred
            df_res['Residual'] = y_pred - y_true
            df_res.to_csv(os.path.join(model_folder, f"{model_id}_Predictions.csv"), index=False)

            # Append to Master
            res_row = {
                'Model_ID': model_id, 'Inputs': inp_set, 'Arch': layers, 'Act': act,
                'RMSE': rmse, 'R2': r2, 'MBE': mbe, 'MSE': mse, 'NSE': nse, 'd_index': d,
                'Slope': slope, 'Intercept': intercept
            }
            master_results.append(res_row)
            pd.DataFrame(master_results).to_csv(master_log_path, index=False)

            tf.keras.backend.clear_session()

print(f"\n*** ALL MODELS COMPLETE ***\nResults at: {RUN_PATH}")
