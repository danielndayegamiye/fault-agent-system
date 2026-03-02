# Fault Detection and Diagnostic Pipeline

This repository provides an end-to-end machine learning pipeline for detecting electrical faults and diagnosing their specific types. 

It uses XGBoost models to process electrical features and classify system states.

## Project Structure

* `src/data/loader.py`: Prepares dataset splits, handles missing values, and scales features.
* `src/agents/detection_agent.py`: A binary classifier that determines if a fault is present (0 for normal, 1 for fault).
* `src/agents/diagnostic_agent.py`: A multi-class classifier that identifies the specific fault type.
* `run_pipeline.py`: The main command-line script to train the agents, evaluate them, and save feature importances.

## Training the Models

First, ensure you have the required dependencies installed: 
`pip install numpy pandas scikit-learn xgboost`

Run the pipeline from the terminal by pointing it to your data files and specifying where to save the outputs. 

```bash
python run_pipeline.py \
    --detection data/detect_dataset.csv \
    --diagnostic data/classData.csv \
    --save-dir models/ \
    --n-estimators 300 \
    --learning-rate 0.05

```

The script will train the models, print evaluation metrics (like accuracy and ROC-AUC) to the console, and save the `.pkl` model files alongside feature importance text files in your specified directory.

## Using the Models on New Data

Once trained, you can load the saved models to run predictions on new electrical data. The input data must be a 2D numpy array containing the six electrical features (Ia, Ib, Ic, Va, Vb, Vc).

```python
import numpy as np
from src.agents.detection_agent import DetectionAgent
from src.agents.diagnostic_agent import DiagnosticAgent

# Load the saved models from disk
det_agent = DetectionAgent.load("models/detection_agent.pkl")
diag_agent = DiagnosticAgent.load("models/diagnostic_agent.pkl")

# Prepare new data as a 2D numpy array (Ia, Ib, Ic, Va, Vb, Vc)
new_data = np.array([[10.5, -5.2, -4.8, 220.1, -110.0, -109.5]])

# Get binary detection prediction (0 = normal, 1 = fault)
is_fault = det_agent.predict(new_data)

# Get multi-class diagnostic prediction as a human-readable string
fault_type = diag_agent.predict_labels(new_data)

print(f"Fault detected: {is_fault[0]}")
print(f"Fault type: {fault_type[0]}")

