from __future__ import annotations

import importlib
import ipaddress
import logging
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.dataset import PROJECT_ROOT, TARGET_COLUMN, load_reference_dataset
from src.logging_config import configure_logging
from src.metrics import evaluate_predictions
from src.model_registry import MODEL_DEFINITIONS
from src.preprocessing import build_pipeline

LOGGER = logging.getLogger("ml_practice.app")
MAX_REMOTE_DATASET_BYTES = 10 * 1024 * 1024
RANDOM_STATE = 42

st.set_page_config(
    page_title="Model Lab | Classification",
    page_icon="📊",
    layout="wide",
)

LOGGER = configure_logging()


def validate_labeled_data(
    dataset: pd.DataFrame,
    target_column: str,
    expected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if dataset.empty:
        raise ValueError("The CSV contains no data rows.")
    if dataset.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    if target_column not in dataset.columns:
        raise ValueError(f"Target column {target_column!r} is missing from the CSV.")
    if dataset[target_column].isna().any():
        raise ValueError("The target column contains missing values.")

    features = dataset.drop(columns=target_column)
    if features.empty:
        raise ValueError("The CSV must contain at least one feature column.")
    if expected_features is not None:
        missing = [column for column in expected_features if column not in features.columns]
        extra = [column for column in features.columns if column not in expected_features]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing features: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected columns: {', '.join(extra)}")
            raise ValueError("Uploaded data does not match the saved models (" + "; ".join(details) + ").")
        features = features.loc[:, expected_features]

    target = dataset[target_column]
    if target.nunique(dropna=False) < 2:
        raise ValueError("Classification evaluation requires at least two target classes.")
    if len(dataset) < 10:
        raise ValueError("Provide at least 10 labeled rows for a stable evaluation.")
    return features, target



st.title("Classification Model Lab")
st.markdown(
    "Compare five classifiers with a held-out test set. Explore the committed "
    "Wisconsin models or run a separate experiment on your own public CSV."
)

mode = st.sidebar.radio(
    "Choose a workspace",
    ("Saved reference models", "Experiment with a dataset URL", "Deployment setup"),
)

if mode == "Saved reference models":
    st.sidebar.caption("Pre-trained models are loaded as-is; uploaded test data is never used for training.")
    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    try:
        if uploaded_file is None:
            test_path = PROJECT_ROOT / "test_data.csv"
            if not test_path.is_file():
                raise FileNotFoundError("test_data.csv is missing from the repository.")
            evaluation_data = pd.read_csv(test_path)
            st.caption("Showing the committed, stratified 20% holdout by default.")
        else:
            evaluation_data = pd.read_csv(uploaded_file)
        show_results(evaluation_data, TARGET_COLUMN, reference=True)
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
        LOGGER.exception("Reference model evaluation failed")
        st.error(str(error))
elif mode == "Experiment with a dataset URL":
    st.sidebar.caption(
        "This workflow fits models on 80% of the supplied dataset and reports metrics "
        "on its stratified 20% holdout. It does not change the saved reference models."
    )
    dataset_url = st.text_input(
        "Public HTTPS URL to a direct CSV",
        placeholder="https://example.org/classification-data.csv",
    )
    if st.button("Load and evaluate dataset", type="primary"):
        if not dataset_url.strip():
            st.error("Enter a public HTTPS URL to a direct CSV file.")
        else:
            try:
                st.session_state["remote_dataset"] = read_public_csv(dataset_url)
                st.session_state["remote_dataset_url"] = dataset_url.strip()
                LOGGER.info("Downloaded a public CSV with %d rows", len(st.session_state["remote_dataset"]))
            except (ValueError, requests.RequestException, pd.errors.ParserError) as error:
                LOGGER.exception("Dataset URL experiment failed")
                st.error(str(error))

    remote_data = st.session_state.get("remote_dataset")
    if remote_data is not None:
        st.success(
            f"Loaded {len(remote_data):,} rows and {len(remote_data.columns)} columns "
            f"from {st.session_state['remote_dataset_url']}."
        )
        target_column = st.selectbox(
            "Choose the classification target",
            list(remote_data.columns),
            index=len(remote_data.columns) - 1,
        )
        if st.button("Run generic experiment", type="primary"):
            try:
                show_results(remote_data, target_column, reference=False)
            except (ValueError, pd.errors.ParserError) as error:
                LOGGER.exception("Dataset URL experiment failed")
                st.error(str(error))
else:
    st.subheader("Streamlit Community Cloud deployment setup")
    st.markdown(
        "Check the repository settings before opening Community Cloud. GitHub "
        "authorization and the final deploy action must happen in your own account."
    )
    with st.form("deployment_settings"):
        repository_url = st.text_input(
            "GitHub repository URL",
            placeholder="https://github.com/owner/repository",
        )
        deployment_branch = st.text_input("Branch", value="main")
        entrypoint = st.text_input("App entry point", value="app.py")
        submitted = st.form_submit_button("Check deployment settings")

    if submitted:
        parsed_repository = urlparse(repository_url.strip())
        repository_parts = [
            part for part in parsed_repository.path.strip("/").split("/") if part
        ]
        if (
            parsed_repository.scheme != "https"
            or parsed_repository.netloc.lower() != "github.com"
            or len(repository_parts) != 2
            or not repository_parts[0]
            or not repository_parts[1].removesuffix(".git")
        ):
            st.error("Enter a repository URL in the form https://github.com/owner/repository.")
        elif not deployment_branch.strip():
            st.error("Enter the branch that contains the app.")
        else:
            relative_entrypoint = Path(entrypoint.replace("\\", "/"))
            if (
                relative_entrypoint.is_absolute()
                or ".." in relative_entrypoint.parts
                or relative_entrypoint.suffix.lower() != ".py"
                or not (PROJECT_ROOT / relative_entrypoint).is_file()
            ):
                st.error("The app entry point must be a Python file present in this project.")
            else:
                LOGGER.info(
                    "Validated deployment configuration for %s on branch %s",
                    parsed_repository.netloc,
                    deployment_branch.strip(),
                )
                st.success("The repository URL, branch, and app entry point are ready.")
                st.code(
                    f"Repository: {repository_url.strip()}\n"
                    f"Branch: {deployment_branch.strip()}\n"
                    f"Main file: {relative_entrypoint.as_posix()}"
                )
                st.link_button(
                    "Continue to Streamlit Community Cloud",
                    "https://share.streamlit.io/",
                    type="primary",
                )
                st.info(
                    "Community Cloud requires a signed-in GitHub account and an "
                    "explicit deployment confirmation. No passwords or tokens are "
                    "requested or sent by this application."
                )