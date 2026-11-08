# Classification Model Lab

## a. Problem statement

This project compares five supervised classification models for the Wisconsin
Diagnostic Breast Cancer dataset. The target indicates whether a breast mass
is malignant (`M`) or benign (`B`). The Streamlit app demonstrates model
evaluation on held-out labeled data and allows separate experiments using
public CSV datasets.

This project is an educational example, not a medical diagnostic tool.

## b. Dataset description

The reference dataset is the **Breast Cancer Wisconsin (Diagnostic)** dataset
from the UCI Machine Learning Repository.

- **Instances:** 569
- **Model features:** 30 numeric measurements
- **Target:** `Diagnosis` (`M` = malignant, `B` = benign)
- **Class counts:** 212 malignant and 357 benign
- **Excluded field:** `ID` is retained in the source data for provenance but
  removed before model training.
- **Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin
  (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Included source CSV:** `data/breast_cancer_wisconsin_diagnostic.csv`
- **Held-out test CSV:** `test_data.csv`

The 30 model features are ten measurements—radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, and fractal
dimension—recorded as mean, standard error, and worst values.


## Live Streamlit app

The app is **deployed ** at [Web-App] (https://gaurav-patil-ml-assignment-2.streamlit.app/) , so there is no live app URL to share. See
[Streamlit Community Cloud deployment](#streamlit-community-cloud-deployment)
for the manual deployment steps.

## Run locally

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train_models
streamlit run app.py
```

If PowerShell script activation is disabled, use Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m src.train_models
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## d. Models and evaluation

The five reference models use the same stratified 80/20 split (random seed 69).
This yields 455 training rows and 114 held-out test rows. Preprocessing is
fitted within training folds. Logistic Regression, Decision Tree, K-Nearest
Neighbors, and Support Vector Machine are tuned using `GridSearchCV`; Gaussian
Naive Bayes uses its standard configuration. Hyperparameter selection and
cross-validation use only the training portion. The untouched holdout is used
for the final comparison.

The table reports holdout metrics. Precision, Recall, and F1 are macro
averages, so Recall is not a duplicate of Accuracy. `Malignant Recall (M)`
reports the fraction of malignant holdout cases correctly identified.

| ML Model Name | Accuracy | AUC | Precision (macro) | Recall (macro) | F1 (macro) | MCC | Malignant Recall (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.974 | 0.995 | 0.974 | 0.969 | 0.972 | 0.943 | 0.952 |
| Decision Tree | 0.912 | 0.945 | 0.909 | 0.901 | 0.905 | 0.810 | 0.857 |
| K-Nearest Neighbors | 0.939 | 0.982 | 0.948 | 0.922 | 0.932 | 0.869 | 0.857 |
| Gaussian Naive Bayes | 0.921 | 0.989 | 0.922 | 0.908 | 0.914 | 0.829 | 0.857 |
| Support Vector Machine | 0.974 | 0.993 | 0.980 | 0.964 | 0.971 | 0.944 | 0.929 |


### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Tied for top holdout accuracy (111 of 114 correct) and identified 40 of 42 malignant cases, missing two. |
| Decision Tree | The most interpretable model, but it missed six malignant cases and had the lowest holdout MCC. |
| K-Nearest Neighbors | Strong cross-validation accuracy, but it missed six malignant cases on the holdout. |
| Gaussian Naive Bayes | Had competitive AUC, but missed six malignant cases on the holdout. |
| Support Vector Machine | Also classified 111 of 114 rows correctly and had the highest holdout MCC by a very small margin; it missed three malignant cases. |
| Overall winner | Logistic Regression is the preferred model for this evaluation because malignant recall was the primary selection criterion, and it identified the most malignant cases on the holdout (40/42; 95.2%). It tied SVM on accuracy and missed one fewer malignant case. This is a criterion-based choice on one holdout—not evidence of statistically significant superiority or clinical readiness. |

For Logistic Regression, the confusion matrix contains two false negatives
(malignant cases predicted benign) and one false positive. For SVM, it contains
three false negatives and no false positives. The app displays the selected
model's confusion matrix and classification report, and highlights malignant
false negatives for the reference dataset.

## App features

- Score saved reference models against the included or uploaded labeled test
  CSV without retraining. The original source `ID` column is ignored if present.
- Choose a model and view its metrics, confusion matrix, and classification
  report.
- Run training and holdout experiments on a public CSV URL, choose an eligible
  target column, and select which models to train.
- Generate per-model performance observations and select a preferred generic
  experiment model by the highest mean cross-validation macro-F1. The report
  also shows the preferred model's holdout scores and clarifies that this
  criterion does not establish statistical significance.
- See progress, elapsed time, and an approximate remaining-time estimate while
  experiments run.
- Open **User Guide — How to use** in the app for in-app instructions.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with an
   authorized GitHub account.
3. Create an app using this repository, the desired branch, and `app.py`.
4. Confirm deployment in the Streamlit interface.

Deployment requires account authorization and confirmation. This project does
not collect credentials or deploy itself in the background.

## Logging

Application events are written to `logs/app.log` and the console. The active
log rotates at local midnight or 30 MiB, whichever occurs first, and retains up
to seven rotated files.
