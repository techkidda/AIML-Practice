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


def read_public_csv(url: str) -> pd.DataFrame:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a public HTTPS URL to a direct CSV file.")

    try:
        resolved_addresses = {
            entry[4][0]
            for entry in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise ValueError("The dataset host could not be resolved.") from error
    if not resolved_addresses or any(
        not ipaddress.ip_address(address).is_global for address in resolved_addresses
    ):
        raise ValueError("The dataset URL must resolve only to public IP addresses.")

    try:
        with requests.get(
            url.strip(),
            timeout=(5, 20),
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "ClassificationModelLab/1.0"},
        ) as response:
            if response.is_redirect:
                raise ValueError("Redirecting dataset URLs are not supported; use the final CSV URL.")
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_REMOTE_DATASET_BYTES:
                raise ValueError("The CSV is larger than the 10 MiB download limit.")
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                payload.extend(chunk)
                if len(payload) > MAX_REMOTE_DATASET_BYTES:
                    raise ValueError("The CSV is larger than the 10 MiB download limit.")
    except requests.RequestException as error:
        raise ValueError(f"Could not download the dataset: {error}") from error

    try:
        return pd.read_csv(BytesIO(payload))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("The URL did not return a readable CSV file.") from error


@st.cache_resource
def load_saved_model(slug: str, modified_ns: int):
    del modified_ns
    artifact_path = PROJECT_ROOT / "models" / f"{slug}.joblib"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Saved artifact {artifact_path.name} is missing. Run `python -m src.train_models`."
        )
    return joblib.load(artifact_path)


def evaluate_all(
    dataset: pd.DataFrame,
    target_column: str,
    reference: bool,
) -> tuple[pd.DataFrame, dict[str, tuple[object, object, list[object]]]]:
    expected_features = (
        load_reference_dataset().drop(columns=TARGET_COLUMN).columns.tolist()
        if reference
        else None
    )
    features, target = validate_labeled_data(dataset, target_column, expected_features)
    if reference:
        evaluation_features, evaluation_target = features, target
        x_train = y_train = None
    else:
        class_counts = target.value_counts()
        if class_counts.min() < 3:
            raise ValueError(
                "For a stratified train/holdout experiment, each target class needs "
                "at least 3 rows."
            )
        x_train, evaluation_features, y_train, evaluation_target = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )

    rows = []
    details: dict[str, tuple[object, object, list[object]]] = {}
    for definition in MODEL_DEFINITIONS:
        if reference:
            artifact_path = PROJECT_ROOT / "models" / f"{definition.slug}.joblib"
            model = load_saved_model(definition.slug, artifact_path.stat().st_mtime_ns)
            unseen_classes = set(evaluation_target.unique()) - set(model.classes_)
            if unseen_classes:
                raise ValueError(
                    "Uploaded target contains classes not present in the saved models: "
                    + ", ".join(sorted(map(str, unseen_classes)))
                )
        else:
            module = importlib.import_module(definition.module)
            classifier = getattr(module, definition.class_name)().build()
            model = build_pipeline(classifier, x_train).fit(x_train, y_train)

        model_metrics, matrix, labels = evaluate_predictions(
            model, evaluation_features, evaluation_target
        )
        rows.append({"ML Model Name": definition.display_name, **model_metrics})
        details[definition.display_name] = (model, matrix, labels)
    return pd.DataFrame(rows), details


def show_results(
    dataset: pd.DataFrame,
    target_column: str,
    reference: bool,
) -> None:
    results, details = evaluate_all(dataset, target_column, reference)
    evaluated_rows = int(next(iter(details.values()))[1].sum())
    LOGGER.info(
        "Evaluated %d models on %d holdout rows from a %d-row dataset (reference=%s)",
        len(details),
        evaluated_rows,
        len(dataset),
        reference,
    )
    st.subheader("Model comparison")
    st.dataframe(
        results.style.format(
            {name: "{:.3f}" for name in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")},
            na_rep="N/A",
        ),
        width="stretch",
        hide_index=True,
    )

    selected_name = st.selectbox("Inspect a model", list(details))
    model, matrix, labels = details[selected_name]
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {selected_name}: confusion matrix")
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=[str(label) for label in labels],
            yticklabels=[str(label) for label in labels],
            xlabel="Predicted label",
            ylabel="True label",
        )
        axis.tick_params(axis="x", labelrotation=25)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.markdown(f"#### {selected_name}: classification report")
        model_features, model_target = validate_labeled_data(dataset, target_column)
        if not reference:
            _, model_features, _, model_target = train_test_split(
                model_features,
                model_target,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=model_target,
            )
        predictions = model.predict(model_features)
        st.code(
            classification_report(
                model_target,
                predictions,
                labels=labels,
                zero_division=0,
            ),
            language="text",
        )


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
