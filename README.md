# Classification Models: Wisconsin Diagnostic Dataset

## Problem statement
Compare five supervised classification algorithms for predicting whether a
breast mass is malignant or benign, and provide an interactive way to evaluate
the saved models on labeled test data.

## Dataset description
The Breast Cancer Wisconsin (Diagnostic) dataset is published by the UCI
Machine Learning Repository. It contains 569 instances, 30 numeric diagnostic
features, and a binary diagnosis target (`M` = malignant, `B` = benign). The
record identifier is retained in the source CSV for provenance and excluded
from model features. The held-out test data is stratified and contains no
training rows.

- Source: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- CSV: `data/breast_cancer_wisconsin_diagnostic.csv`
- Test data: `test_data.csv`

## GitHub Repository Link

Repository URL: https://github.com/techkidda/AIML-Practice

## Live Streamlit App Link

The app has not been deployed yet. After pushing this repository, deploy
`app.py` from <https://share.streamlit.io/> using the steps below.

## Execution screenshot

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m src.train_models
streamlit run app.py
```

Upload a labeled CSV with the same feature columns and target column to evaluate
the saved reference models. The app also supports a public HTTPS CSV URL for a
separate train/holdout experiment on a user-selected dataset.
Open **User Guide — How to use** in the app's workspace selector for in-app
instructions covering both workflows, valid targets, the train/test split, and
how to interpret the evaluation metrics.

## Models used

All models use the same stratified 80/20 split (random seed 42). Preprocessing
is fitted on training data only and packaged with each saved model.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.965 | 0.996 | 0.965 | 0.965 | 0.965 | 0.925 |
| Decision Tree | 0.904 | 0.947 | 0.907 | 0.904 | 0.901 | 0.793 |
| K-Nearest Neighbors | 0.956 | 0.983 | 0.957 | 0.956 | 0.956 | 0.906 |
| Naive Bayes | 0.921 | 0.989 | 0.921 | 0.921 | 0.920 | 0.829 |
| Support Vector Machine | 0.965 | 0.995 | 0.965 | 0.965 | 0.965 | 0.925 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline; it produced the highest AUC (0.996), indicating excellent ranking of the two classes. |
| Decision Tree | Easiest to interpret, but it had the lowest test accuracy and MCC, suggesting weaker generalization on this split. |
| K-Nearest Neighbors | Strong test performance after feature scaling, with accuracy above 95% and balanced weighted precision/recall. |
| Naive Bayes | Competitive AUC but lower accuracy than the top models; its conditional-independence assumption may limit classification quality. |
| Support Vector Machine | Tied for top test accuracy and marginally ahead on F1 and MCC; it remains a strong, well-balanced classifier on this split. |
| Overall, Winner for your dataset? | Support Vector Machine by a small F1/MCC margin. Logistic Regression has the highest AUC (0.996), and both achieve 0.965 accuracy. |

## Streamlit Community Cloud deployment

Push this repository to GitHub, sign in at
<https://share.streamlit.io/>, choose **Create app**, select this repository,
branch, and `app.py`, then deploy. Streamlit Community Cloud currently requires
this account-authorized flow; the app does not collect credentials or attempt
to provision a deployment in the background.

## Logging

Application events are written to `logs/app.log` and the console. The active
log rotates at local midnight or 30 MiB, whichever occurs first, and retains
up to seven rotated files.


## Create venv

Bash
### 1. Install Python 3 and venv support (if not already installed)
sudo dnf install -y python3 python3-pip

### 2. Navigate to your project directory
cd /path/to/your/project

### 3. Create the virtual environment (named 'venv')
python3 -m venv venv

### 4. Activate the virtual environment
source venv/bin/activate

### 5. Verify activation (should point to project's venv)
which python

### 6. Deactivate when finished
deactivate

## Windows
### 1. Install Python 3 and venv support (if not already installed)
sudo dnf install -y python3 python3-pip

### 2. Navigate to your project directory
cd /path/to/your/project

### 3. Create the virtual environment (named 'venv')
python3 -m venv venv

### 4. Activate the virtual environment
source venv/bin/activate

### 5. Verify activation (should point to project's venv)
which python

### 6. Deactivate when finished
deactivate


## Bash

### 1. Navigate to your project directory
cd /path/to/your/project

### 2. Create the virtual environment
python3 -m venv venv

### 3. Activate the virtual environment
source venv/bin/activate

### 4. Verify activation
which python

### 5. Deactivate when finished
deactivate