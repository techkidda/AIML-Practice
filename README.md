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
