**Healthcare Monitoring with ARIMA Forecasting**
 
This repository contains a Python-based project for real-time healthcare monitoring, focusing on forecasting patient heart rates using the ARIMA (AutoRegressive Integrated Moving Average) model. The project leverages a dataset of patient vitals to predict short-term heart rate trends, enabling potential anomaly detection and continuous monitoring in healthcare settings.
**Project Overview**
The project processes a dataset (healthcare_monitoring_dataset.csv) containing patient vitals such as heart rate, blood pressure, SpO2, and more, collected on May 17–18, 2025. The primary goal is to forecast heart rate for individual patients using time series analysis, achieving a Mean Absolute Error (MAE) of ~2.5 BPM for short-term predictions. Key features include:

Data Preprocessing: Handles irregular timestamps via resampling to 1-minute intervals and interpolation.
ARIMA Modeling: Fits an ARIMA(5,d,1) model with adaptive differencing based on stationarity tests.
Real-Time Forecasting: Generates 10-minute heart rate predictions for continuous monitoring.
Error Handling: Robust checks for insufficient or constant data to ensure reliable analysis.
Visualization: Plots training, actual, and forecasted heart rates, saved as arima_heart_rate_forecast.png.

**Dataset**
The dataset (healthcare_monitoring_dataset.csv) includes the following columns:

Timestamp: Datetime (e.g., 2025-05-18 07:55:44.594978).
Patient ID: Unique identifiers (e.g., PAT1003, PAT1094).
Heart Rate: Beats per minute (BPM), numeric.
Blood Pressure, Blood Oxygen (SpO2), Respiratory Rate, Glucose Level, Temperature, ECG Signal, Activity Level, Label, Device ID: Additional vitals and metadata.

The dataset spans ~1 day with ~500 records, though patient-specific data (e.g., PAT1003) may be sparse, requiring resampling for time series analysis.
**Installation
Prerequisites**

Python 3.8 or higher
pip for package management

Setup

**Clone the Repository:**
git clone https://github.com/your-username/Healthcare-Monitoring-ARIMA.git
cd Healthcare-Monitoring-ARIMA


**Install Dependencies:**
pip install pandas numpy matplotlib statsmodels scikit-learn


**Prepare the Dataset:**

Place healthcare_monitoring_dataset.csv in the project root directory.
Ensure the file matches the expected format (see Dataset).



**Usage**

**Run the Analysis:**

Execute the main script to perform ARIMA forecasting for a patient (default: PAT1003):python arima_healthcare_monitoring.py


To analyze a different patient, modify patient_id in arima_healthcare_monitoring.py (e.g., patient_id = 'PAT1094').


**Outputs:**

Console: Displays data points, ADF test p-value, MAE, 10-minute forecast, and insights.
Files:
arima_heart_rate_forecast.png: Plot of training, actual, and forecasted heart rates.
arima_model.pkl: Saved ARIMA model for reuse.




**Sample Output:**
Initial Heart Rate data points for PAT1003: 3
Resampled Heart Rate data points: 22
ADF p-value: 0.1234
Mean Absolute Error: 2.50 BPM

Future 10-Minute Forecast:
                       Forecasted Heart Rate
2025-05-18 07:33:44.594978          75.23
...

**Insights from ARIMA Analysis:**
1. Model Fit: The ARIMA(5,1,1) model achieved an MAE of 2.50 BPM, indicating good accuracy.
2. Stationarity: The Heart Rate series required 1 order differencing, suggesting trending behavior.
...



**Project Structure**
Healthcare-Monitoring-ARIMA/
├── arima_healthcare_monitoring.py  # Main script for ARIMA forecasting
├── healthcare_monitoring_dataset.csv  # Input dataset (not tracked)
├── arima_heart_rate_forecast.png  # Output plot (generated)
├── arima_model.pkl  # Saved ARIMA model (generated)
└── README.md  # This file

**Future Improvements**

Multi-Patient Analysis: Extend the model to process multiple patients simultaneously.
Additional Vitals: Forecast other metrics like glucose levels or SpO2.
Anomaly Detection: Implement thresholds for detecting abnormal heart rates (e.g., >100 BPM).
Real-Time Integration: Deploy the model in a live monitoring system using streaming data.
Hyperparameter Tuning: Optimize ARIMA parameters (p,d,q) using grid search.

**Contributing
Contributions are welcome! Please:**

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.


GitHub Issues: https://github.com/ARUNAGIRINATHAN-K/Healthcare-Monitoring-ARIMA/issues
LinkedIn: www.linkedin.com/in/arunagirinathan-k


Happy forecasting! 🚀
