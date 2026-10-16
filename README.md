# Network Traffic Analyzer & Anomaly Detection Platform

An end-to-end machine learning platform for analyzing network traffic, detecting anomalous flows, classifying traffic, assigning risk scores, and presenting security insights through an interactive Streamlit dashboard.

---

## Overview

This project combines network traffic analysis, supervised machine learning, unsupervised anomaly detection, risk scoring, and interactive visualization.

### System Pipeline

Network Traffic Dataset
→ Data Preprocessing
→ Feature Engineering
→ Random Forest Classification
→ Isolation Forest Anomaly Detection
→ Risk Scoring
→ Interactive Dashboard

---

## Key Features

- Network traffic data ingestion
- Data cleaning and preprocessing
- Automatic handling of missing values
- Automatic handling of infinite values
- Network traffic feature engineering
- Random Forest traffic classification
- Isolation Forest anomaly detection
- Anomaly score generation
- 0–100 traffic risk scoring
- Low / Medium / High risk categorization
- Suspicious flow identification
- Interactive Streamlit dashboard
- Traffic distribution analysis
- Feature-level traffic analysis
- Anomaly visualization
- Risk analysis
- High-risk flow analysis
- Automated regression testing

---

## Machine Learning

### 1. Random Forest Traffic Classification

A Random Forest classifier is used for supervised network traffic classification.

The current demonstration uses a balanced sample from the CIC-IDS2017 DDoS traffic dataset:

- 1,000 BENIGN flows
- 1,000 DDoS flows
- Total: 2,000 flows
- 80/20 stratified train-test split
- 400 test samples

### Validation Result

The current balanced sample validation achieved:

- Accuracy: 100%
- Test samples: 400

Confusion Matrix:

    [[200   0]
     [  0 200]]

The model correctly classified all 400 flows in this specific held-out sample.

> Important: This result represents the demonstrated 2,000-flow sample only. It should not be interpreted as general performance across the complete CIC-IDS2017 dataset.

---

### 2. Isolation Forest Anomaly Detection

Isolation Forest is used for unsupervised anomaly detection.

Unlike supervised classification, anomaly detection attempts to identify unusual traffic patterns without directly using attack labels during training.

The system provides:

- Normal traffic detection
- Anomalous traffic detection
- Anomaly scores
- Anomaly distribution visualization

### Real Dataset Validation

A real CIC-IDS2017 sample was successfully processed through the anomaly detection pipeline.

Example validation result:

    Normal:  374
    Anomaly: 26

> Isolation Forest is used here as an anomaly detector, not as a direct attack classifier.

---

### 3. Risk Scoring

The platform combines anomaly information and predicted traffic category to generate a normalized risk score between 0 and 100.

Risk levels:

- Low
- Medium
- High

Example:

    Traffic Flow
         ↓
    Classification
         ↓
    Anomaly Detection
         ↓
    Risk Score
         ↓
    Risk Level

The system can identify potentially high-risk flows for further investigation.

---

## Dashboard

The Streamlit application contains five main sections.

### 1. Overview

Provides:

- Total flows
- BENIGN traffic
- DDoS traffic
- Detected anomalies
- High-risk flows
- Traffic distribution
- Risk distribution
- Random Forest test accuracy

### 2. Traffic Analysis

Provides interactive analysis of numerical network traffic features.

Users can explore:

- Feature distributions
- Traffic measurements
- Protocol distribution when available

### 3. Anomaly Detection

Displays:

- Normal flow count
- Anomaly count
- Anomaly score distribution
- Detected anomalous flows
- Flow-level anomaly information

### 4. Risk Analysis

Displays:

- Low-risk traffic
- Medium-risk traffic
- High-risk traffic
- Risk distribution
- High-risk flow details

### 5. Suspicious Flows

Suspicious flows are identified when:

- The predicted traffic category is not BENIGN, or
- The flow is classified as anomalous

The dashboard provides:

- Suspicious flow count
- Detailed suspicious-flow table
- Highest-risk flows

---

## End-to-End Pipeline Validation

The complete pipeline was validated using a balanced sample of CIC-IDS2017 traffic.

### Dataset

    BENIGN: 1,000
    DDoS:   1,000
    Total:  2,000

### Train/Test Split

    Training: 1,600
    Testing:    400

### Random Forest Classification

    Accuracy: 1.00

    Confusion Matrix:

    [[200   0]
     [  0 200]]

### Isolation Forest

    Normal:  374
    Anomaly:  26

### Risk Distribution

    Low:     155
    Medium:  241
    High:      4

### Suspicious Flows

    Suspicious flows identified: 226

These results demonstrate that the complete preprocessing → feature engineering → classification → anomaly detection → risk scoring → dashboard pipeline executes successfully on real network traffic data.

> The reported classification result is specific to the demonstrated balanced sample and should not be interpreted as general model performance.

---

## Testing

The project includes automated tests covering the main data-processing, feature-engineering, machine-learning, and risk-scoring components.

### Test Files

    tests/
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_detection.py
    ├── test_classification.py
    └── test_risk_scoring.py

### Tested Components

#### Preprocessing

- Column-name normalization
- Label normalization
- Feature/label separation
- Numeric conversion
- Missing-value handling
- Infinite-value handling

#### Feature Engineering

- Total packet calculation
- Forward packet ratio
- Total byte calculation
- Forward byte ratio
- Average packet size
- Duration conversion

#### Anomaly Detection

- Numeric feature preparation
- Missing-value imputation
- Isolation Forest training
- Anomaly prediction
- Anomaly score generation
- Invalid contamination handling
- Minimum-row validation

#### Classification

- Numeric feature preparation
- Random Forest training
- Multi-class validation
- Feature/label length validation
- Minimum-class validation
- Prediction generation

#### Risk Scoring

- Risk score calculation
- Score normalization
- Attack-category component
- Risk-level assignment
- Invalid score validation
- Input-length validation

### Final Regression Test

The complete project test suite was executed successfully.

    26 passed in 3.76s

Final status:

    26 tests
    26 passed
    0 failed

Run the complete test suite with:

    pytest -q

---

## Dataset

This project uses the publicly available CIC-IDS2017 dataset.

### Dataset Source

Canadian Institute for Cybersecurity — Intrusion Detection Systems Dataset 2017

Official source:

https://www.unb.ca/cic/datasets/ids-2017.html

The full dataset is not included in this repository because of its large size.

---

## Project Structure

    Network-Traffic-Analyzer/
    │
    ├── app/
    │   └── app.py
    │
    ├── data/
    │   └── raw/
    │
    ├── models/
    │
    ├── screenshots/
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── feature_engineering.py
    │   ├── anomaly_detection.py
    │   ├── classification.py
    │   └── risk_scoring.py
    │
    ├── tests/
    │   ├── test_preprocessing.py
    │   ├── test_features.py
    │   ├── test_detection.py
    │   ├── test_classification.py
    │   └── test_risk_scoring.py
    │
    ├── README.md
    ├── requirements.txt
    ├── pytest.ini
    ├── .gitignore
    └── LICENSE

---

## Technologies

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest
- Isolation Forest
- SimpleImputer
- Pipeline

### Visualization and Dashboard

- Streamlit
- Plotly

### Testing

- Pytest

### Dataset

- CIC-IDS2017

---

## Installation

### 1. Clone the Repository

    git clone git@github.com:mdsakibhasan-cs/Network-Traffic-Analyzer.git
    cd Network-Traffic-Analyzer

### 2. Create a Virtual Environment

    python -m venv .venv

### 3. Activate the Environment on Windows

    .\.venv\Scripts\Activate.ps1

### 4. Install Dependencies

    pip install -r requirements.txt

### 5. Run the Dashboard

    streamlit run app/app.py

### 6. Run Tests

    pytest -q

---

## Requirements

Main dependencies:

    pandas
    numpy
    scikit-learn
    streamlit
    plotly
    pytest

All dependencies are listed in requirements.txt.

---

## Security and Data Considerations

This project analyzes recorded network traffic datasets for research and educational purposes.

It does not perform active network attacks or automated exploitation.

The system is designed for:

- Network traffic analysis
- Anomaly detection
- Security analytics
- Machine learning experimentation
- Educational cybersecurity research

---

## Limitations

The current implementation has several limitations:

1. The demonstrated classification result is based on a balanced 2,000-flow sample.
2. The current supervised classifier focuses on the demonstrated BENIGN/DDoS classification scenario.
3. Isolation Forest detects unusual patterns but does not inherently identify the exact attack type.
4. The current risk score is a heuristic combination of anomaly and classification signals.
5. The complete CIC-IDS2017 dataset is not included in the repository.
6. Model performance may change with different datasets, attack classes, sampling strategies, and feature distributions.
7. The current system analyzes recorded traffic data rather than live network traffic.

---

## Future Improvements

Possible future extensions include:

- Multi-class attack classification
- Additional CIC-IDS2017 attack categories
- SHAP-based model explainability
- More advanced risk reasoning
- Real-time traffic ingestion
- Streaming anomaly detection
- Additional machine-learning model comparisons
- Model persistence and loading
- Advanced dashboard filtering
- Time-series traffic analysis
- Cloud deployment
- Automated model evaluation across multiple datasets

---

## Project Status

**Completed — GitHub Ready**

The current implementation includes:

- Data preprocessing
- Feature engineering
- Supervised traffic classification
- Unsupervised anomaly detection
- Risk scoring
- End-to-end pipeline validation
- Interactive Streamlit dashboard
- Automated regression testing
- GitHub repository setup

Final regression status:

    26 passed in 3.76s

---

## Author

**Hasan Md Sakib**

Computer Science and Technology  
Nantong University

GitHub:

https://github.com/mdsakibhasan-cs

---

## Repository

https://github.com/mdsakibhasan-cs/Network-Traffic-Analyzer
