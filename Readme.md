Estimation of Hourly ETo Using Neural Networks for the CIMIS📖 
Overview
This repository contains a complete, two-phase machine learning pipeline for estimating hourly Reference Evapotranspiration ($ET_o$) using data from the California Irrigation Management Information System (CIMIS).The Training Engine: A rigorous grid-search script that evaluates 84 neural network combinations (7 input sets × 4 architectures × 3 activation functions).The Auto-Inference Tool: A standalone graphical user interface (GUI) that allows researchers to apply pre-trained models to new datasets seamlessly, with zero coding required.

→ Key Methodological Features
Zero Data Leakage: A single, universal StandardScaler is fitted exclusively to the training dataset. The inference tool mathematically slices required variables from this universally scaled dataset, ensuring data integrity and fair comparison across all 84 models.
Automated Input Detection: The GUI tool automatically parses the required input variables directly from the .keras filename (e.g., AirTempC_RelHum_Arch16-8-4_tanh.keras), eliminating human error during testing.
Missing Data Handling: The inference tool automatically detects and drops rows containing NaN (blank) values before prediction, preventing matrix calculation failures.

→ Data RequirementsPlease note the specific file formats required for each phase of this pipeline:Training Phase: Expects .parquet files (train.parquet, test.parquet) to handle massive datasets efficiently and preserve column data types.Inference Phase (GUI): Expects standard .csv files for easy integration with new, user-provided datasets.

🛠️ Installation & Prerequisites
To run these scripts, you will need Python 3.8+ and the following libraries installed:
Bash
pip install pandas numpy scikit-learn tensorflow matplotlib pyarrow joblib
(Note: pyarrow is strictly required to read the .parquet training files).

📂 Repository StructurePlaintext📦 Estimation-of-Hourly-ETo
 ┣ 📂 Models                                 # Contains the 7 best trained .keras models
 ┣ 📂 Scalers                                # Contains scaler_X.pkl and scaler_y.pkl
 ┣ 📜 train_grid_search.py                   # Main training engine (requires .parquet data)
 ┣ 📜 run models with test data.py           # Auto-inference GUI application
 ┗ 📜 Test data thermal south hyper arid.csv # Sample CIMIS data for GUI testing

🚀 Usage GuidePhase 

1: Training the Models (train_grid_search.py)This script evaluates models across multiple architectures and activation functions.Open the script and update the BASE_PATH to point to your train.parquet and test.parquet files.
Run the script.The script will generate a new folder containing all 84 trained models, the unified scalers, and a Master_Summary_Report.csv detailing the performance metrics.

2: Running Predictions (run models with test data.py)
To test the pre-trained models on new data using the GUI:Run run models with test data.py. A window will appear.
Click Browse to load your Test Data. Must be a .csv file (e.g., Test data thermal south hyper arid.csv).
Select the specific Trained Model from the Models folder.
Select scaler_X.pkl and scaler_y.pkl from the Scalers folder.
Select results folder.
Click Run Inference & Generate Plot.The tool will automatically clean the data, apply the scaler, slice the correct inputs based on the model's filename, and output the predictions. It will save a Hexbin plot (.png) a results file and a metrics report (.csv) in the folder you chose.

📜 CitationIf you use this code, the single-scaler methodology, or the pre-trained models in your research, please cite the accompanying paper:[Author Last Name, First Initial]. (2026). Estimation of Hourly ETo Using Neural Networks for the CIMIS Network (California) with Various Architectures and Activation Functions. [Computers and Electronics in Agriculture]. [DOI/Link]
