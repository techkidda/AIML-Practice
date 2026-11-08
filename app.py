from __future__ import annotations

import importlib
import ipaddress
import json
import logging
import socket
import time
from collections.abc import Callable
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
from src.model_registry import MODEL_DEFINITIONS, ModelDefinition
from src.public_datasets import (
    CUSTOM_DATASET_OPTION,
    DATASET_BY_LABEL,
    PUBLIC_DATASETS,
    normalize_public_dataset,
)
from src.model_selection import fit_with_cross_validation
from src.progress import estimate_remaining_seconds, format_duration
from src.target_columns import assess_target_columns

LOGGER = logging.getLogger("ml_practice.app")
MAX_REMOTE_DATASET_BYTES = 10 * 1024 * 1024
RANDOM_STATE = 69

st.set_page_config(
    page_title="Model Lab | Classification",
    page_icon="📊",
    layout="wide",
)

LOGGER = configure_logging()


def apply_selected_dataset_url() -> None:
    selected_label = st.session_state["dataset_preset"]
    if selected_label != CUSTOM_DATASET_OPTION:
        st.session_state["dataset_url_input"] = DATASET_BY_LABEL[
            selected_label
        ].url


def validate_labeled_data(
    dataset: pd.DataFrame,
    target_column: str | None,
    expected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if dataset.empty:
        raise ValueError("The CSV contains no data rows.")
    if dataset.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    if not target_column:
        raise ValueError("Select an eligible target column before evaluation.")
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


def load_saved_model_summaries() -> dict[str, dict[str, object]]:
    metrics_path = PROJECT_ROOT / "metrics.json"
    if not metrics_path.is_file():
        return {}
    summaries = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(summaries, dict):
        raise ValueError("metrics.json must contain a JSON object.")
    return summaries


def evaluate_all(
    dataset: pd.DataFrame,
    target_column: str,
    reference: bool,
    progress_callback: Callable[[str, int, str, float | None], None] | None = None,
    model_definitions: tuple[ModelDefinition, ...] = MODEL_DEFINITIONS,
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
    saved_summaries = load_saved_model_summaries() if reference else {}
    for model_index, definition in enumerate(model_definitions, start=1):
        model_started_at = time.perf_counter()
        model_completed = False
        if progress_callback is not None:
            progress_callback("started", model_index, definition.display_name, None)
        try:
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
                model, cross_validation, best_parameters = fit_with_cross_validation(
                    definition.slug, classifier, x_train, y_train
                )

            model_metrics, matrix, labels = evaluate_predictions(
                model, evaluation_features, evaluation_target
            )
            row = {"ML Model Name": definition.display_name, **model_metrics}
            if not reference:
                for metric, scores in cross_validation.items():
                    if metric not in ("Accuracy", "F1", "Malignant Recall (M)"):
                        continue
                    row[f"CV {metric} (mean ± std)"] = (
                        f"{scores['mean']:.3f} ± {scores['std']:.3f}"
                    )
                row["Tuned parameters"] = str(best_parameters)
            else:
                summary = saved_summaries.get(definition.display_name, {})
                if isinstance(summary, dict):
                    cross_validation = summary.get("Cross-validation", {})
                    if isinstance(cross_validation, dict):
                        for metric, scores in cross_validation.items():
                            if metric not in ("Accuracy", "F1", "Malignant Recall (M)"):
                                continue
                            if isinstance(scores, dict) and "mean" in scores and "std" in scores:
                                row[f"CV {metric} (mean ± std)"] = (
                                    f"{scores['mean']:.3f} ± {scores['std']:.3f}"
                                )
                    if "Best parameters" in summary:
                        row["Tuned parameters"] = str(summary["Best parameters"])
            rows.append(row)
            details[definition.display_name] = (model, matrix, labels)
            model_completed = True
        finally:
            if progress_callback is not None:
                progress_callback(
                    "finished" if model_completed else "failed",
                    model_index,
                    definition.display_name,
                    time.perf_counter() - model_started_at,
                )
    return pd.DataFrame(rows), details


def show_results(
    dataset: pd.DataFrame,
    target_column: str,
    reference: bool,
    model_definitions: tuple[ModelDefinition, ...] = MODEL_DEFINITIONS,
    cache_key: tuple[object, ...] | None = None,
) -> None:
    cached_result = st.session_state.get("generic_evaluation_result")
    if (
        not reference
        and cache_key is not None
        and cached_result is not None
        and cached_result[0] == cache_key
    ):
        results, details = cached_result[1], cached_result[2]
    else:
        results = None
        details = None

    total_models = len(model_definitions)
    completed_model_seconds: list[float] = []
    if results is None or details is None:
        evaluation_started_at = time.perf_counter()
        initial_estimate = (
            "Approximate duration: seconds to a few minutes, depending on dataset size "
            "and feature types. The ETA updates after the first model."
        )
        progress_bar = st.progress(0, text=initial_estimate)
        status = st.status("Preparing the train/holdout split...", expanded=True)

        def update_progress(
            stage: str,
            model_index: int,
            model_name: str,
            model_duration: float | None,
        ) -> None:
            elapsed = format_duration(time.perf_counter() - evaluation_started_at)
            if stage == "started":
                estimated_remaining = estimate_remaining_seconds(
                    completed_model_seconds,
                    total_models - model_index + 1,
                )
                eta_text = (
                    f"Approx. {format_duration(estimated_remaining)} remaining "
                    "(including current model)"
                    if estimated_remaining is not None
                    else "ETA estimating from the first model"
                )
                status.update(
                    label=(
                        f"{'Scoring saved' if reference else 'Training and evaluating'} "
                        f"{model_name} ({model_index}/{total_models}) — "
                        f"{eta_text}; elapsed {elapsed}"
                    ),
                    state="running",
                )
                progress_bar.progress(
                    (model_index - 1) / total_models,
                    text=f"{model_name}: in progress · {eta_text} · elapsed {elapsed}",
                )
                return

            if stage == "failed":
                progress_bar.progress(
                    (model_index - 1) / total_models,
                    text=f"{model_name} did not complete · elapsed {elapsed}",
                )
                status.update(
                    label=f"{model_name} failed ({model_index}/{total_models}).",
                    state="error",
                )
                return

            assert model_duration is not None
            completed_model_seconds.append(model_duration)
            estimated_remaining = estimate_remaining_seconds(
                completed_model_seconds,
                total_models - model_index,
            )
            eta_text = (
                "All models complete"
                if estimated_remaining == 0
                else f"Approx. {format_duration(estimated_remaining)} remaining"
            )
            progress_bar.progress(
                model_index / total_models,
                text=f"Completed {model_name} ({model_index}/{total_models}) · {eta_text}",
            )
            status.update(
                label=(
                    f"Completed {model_name} ({model_index}/{total_models}) — "
                    f"{eta_text}; elapsed {elapsed}"
                ),
                state="complete" if model_index == total_models else "running",
            )

        try:
            results, details = evaluate_all(
                dataset,
                target_column,
                reference,
                progress_callback=update_progress,
                model_definitions=model_definitions,
            )
        except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError):
            status.update(label="Evaluation could not be completed.", state="error")
            raise

        evaluated_rows = int(next(iter(details.values()))[1].sum())
        elapsed = format_duration(time.perf_counter() - evaluation_started_at)
        LOGGER.info(
            "Evaluated %d models on %d holdout rows from a %d-row dataset in %s (reference=%s)",
            len(details),
            evaluated_rows,
            len(dataset),
            elapsed,
            reference,
        )
        if not reference and cache_key is not None:
            st.session_state["generic_evaluation_result"] = (
                cache_key,
                results,
                details,
            )

    st.subheader("Model comparison")
    comparison_results = results.drop(columns=["Tuned parameters"], errors="ignore")
    st.dataframe(
        comparison_results.style.format(
            {
                name: "{:.3f}"
                for name in (
                    "Accuracy",
                    "AUC",
                    "Precision",
                    "Recall",
                    "F1",
                    "MCC",
                    "Malignant Recall (M)",
                )
            },
            na_rep="N/A",
        ),
        width="stretch",
        hide_index=True,
    )
    if "Tuned parameters" in results:
        with st.expander("Selected hyperparameters"):
            st.dataframe(
                results[["ML Model Name", "Tuned parameters"]],
                width="stretch",
                hide_index=True,
            )

    selected_name = st.selectbox("Inspect results for a model", list(details))
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
        if "M" in labels:
            malignant_index = labels.index("M")
            false_negatives = int(
                matrix[malignant_index].sum() - matrix[malignant_index, malignant_index]
            )
            st.metric("Malignant cases missed (false negatives)", false_negatives)
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
    "Compare five classifiers using tuned training-fold cross-validation and an "
    "untouched holdout. Explore the committed Wisconsin models or experiment "
    "with a public CSV."
)

mode = st.sidebar.radio(
    "Choose a workspace",
    (
        "User Guide — How to use",
        "Saved reference models",
        "Experiment with a dataset URL",
        "Deployment setup",
    ),
)

if mode == "User Guide — How to use":
    st.title("How to use Classification Model Lab")
    st.markdown(
        "Use this guide to choose a workflow, evaluate the classifiers, and "
        "understand the results."
    )

    st.header("1. Choose an evaluation workflow")
    guide_reference, guide_generic = st.columns(2)
    with guide_reference:
        st.subheader("Saved reference models")
        st.markdown(
            "Use the models already trained on the Wisconsin Breast Cancer dataset. "
            "This workflow **does not train or retrain** models."
        )
        st.markdown(
            "1. Keep the included test data, or upload a labeled CSV.\n"
            "2. For an upload, include the same 30 feature columns and the "
            "`Diagnosis` target (`M` or `B`).\n"
            "3. Compare held-out test metrics and the saved training-fold "
            "cross-validation mean ± std, "
            "then inspect a model's confusion matrix "
            "and classification report."
        )
    with guide_generic:
        st.subheader("Experiment with a dataset URL")
        st.markdown(
            "Train selected classifiers on a new CSV. The five sample UCI datasets "
            "meet the 600-row / 12-feature guideline."
        )
        st.markdown(
            "1. Select a sample dataset, or choose the custom option and paste a "
            "public HTTPS URL to a direct CSV (maximum 10 MiB).\n"
            "2. Click **Load and evaluate dataset**.\n"
            "3. Review eligible target columns and select the target you want to "
            "predict.\n"
            "4. Select one or more classifiers. All five are selected by default.\n"
            "5. Click **Run selected models** to train and compare them."
        )

    st.header("2. How the generic experiment works")
    st.markdown(
        "- The app uses a stratified random **80% training / 20% test split** "
        "with `random_state=69`; the same dataset and target produce the same split.\n"
        "- Grid search and stratified cross-validation run only on the training "
        "portion; the test portion is held back for the final model comparison.\n"
        "- Preprocessing (missing-value imputation, categorical encoding, and "
        "numeric scaling) is fitted using training data only.\n"
        "- Each checked model is trained when you click **Run selected models**. "
        "These experiment models are temporary and do not replace the saved models.\n"
        "- While the run is active, the progress display shows the current model, "
        "elapsed time, and an approximate remaining time. After a run, changing "
        "**Inspect results for a model** reuses its results without training again."
    )

    st.header("3. Choose a valid target")
    st.markdown(
        "The target is the column the classifiers will predict. The app lists "
        "eligible columns with their class counts. A target must have no missing "
        "values, at least two classes, at least three examples in every class, "
        "and fit in the stratified test split. Other columns are explained in the "
        "“Why other columns cannot be targets” expander. Identifier-like or "
        "continuous columns are usually not suitable class labels."
    )

    st.header("4. Read the results")
    st.markdown(
        "| Metric | What it tells you |\n"
        "|---|---|\n"
        "| Accuracy | Fraction of predictions that are correct. |\n"
        "| AUC | How well the model separates/ranks classes; higher is better. |\n"
        "| Precision | How often predictions for a class are correct. |\n"
        "| Recall | Macro-average of how many actual examples of each class the model finds. |\n"
        "| Malignant Recall (M) | For the reference dataset, the fraction of malignant cases found. |\n"
        "| F1 | Macro-average balance between precision and recall. |\n"
        "| MCC | Balanced summary of prediction quality; 1 is perfect, 0 is near "
        "random, and -1 is total disagreement. |"
    )
    st.markdown(
        "Use the confusion matrix to see correct and incorrect predictions by "
        "class. The classification report shows precision, recall, F1, and sample "
        "count for each class. Metrics are measured on the held-out test portion, "
        "not on the training rows. The app also calls out malignant false "
        "negatives (malignant cases predicted benign) for the reference dataset."
    )
    st.info(
        "Saved reference models only score the included or uploaded test data. "
        "Generic URL experiments tune and cross-validate models within the training "
        "split, then score the untouched holdout. Cross-validation mean ± standard "
        "deviation guides model selection; the final holdout is the primary comparison."
    )
elif mode == "Saved reference models":
    st.sidebar.caption(
        "Saved pre-trained models are loaded as-is. This workflow does not retrain them; "
        "uploaded test data is used only for scoring."
    )
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
        "on its stratified 20% holdout. Select the models to fit; all five are selected "
        "by default. This does not change the saved reference models."
    )
    dataset_options = [
        CUSTOM_DATASET_OPTION,
        *(dataset.option_label for dataset in PUBLIC_DATASETS),
    ]
    st.selectbox(
        "Choose a sample classification dataset (each has 12+ features and 600+ rows)",
        dataset_options,
        key="dataset_preset",
        on_change=apply_selected_dataset_url,
    )
    dataset_url = st.text_input(
        "Dataset CSV URL (editable)",
        placeholder="https://example.org/classification-data.csv",
        key="dataset_url_input",
    )
    if st.button("Load and evaluate dataset", type="primary"):
        st.session_state.pop("generic_evaluation_result", None)
        st.session_state.pop("remote_dataset", None)
        st.session_state.pop("remote_dataset_url", None)
        st.session_state.pop("remote_target_column", None)
        if not dataset_url.strip():
            st.error("Enter a public HTTPS URL to a direct CSV file.")
        else:
            try:
                remote_dataset = read_public_csv(dataset_url)
                st.session_state["remote_dataset_url"] = dataset_url.strip()
                matching_dataset = next(
                    (
                        dataset
                        for dataset in PUBLIC_DATASETS
                        if dataset.url == dataset_url.strip()
                    ),
                    None,
                )
                if matching_dataset is not None:
                    remote_dataset = normalize_public_dataset(
                        remote_dataset, dataset_url.strip()
                    )
                st.session_state["remote_dataset"] = remote_dataset
                st.session_state["remote_target_column"] = (
                    matching_dataset.target_column
                    if matching_dataset is not None
                    else None
                )
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
        if remote_data.columns.duplicated().any():
            st.error("This CSV has duplicate column names. Rename them before selecting a target.")
            st.stop()

        assessments = assess_target_columns(remote_data)
        eligible_assessments = [
            assessment for assessment in assessments if assessment.eligible
        ]
        if not eligible_assessments:
            st.error(
                "No column in this dataset can be used as a stratified classification "
                "target. A target must have no missing values, at least two classes, "
                "and at least 3 rows in every class."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": item.column,
                            "Classes": item.classes,
                            "Smallest class": item.minimum_class_samples,
                            "Missing values": item.missing_values,
                            "Eligible target": "No",
                            "Why unavailable": item.reason,
                        }
                        for item in assessments
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.stop()

        st.markdown("#### Eligible target columns")
        st.caption(
            "Choose from these columns only. Eligible targets have no missing values, "
            "at least two classes, and at least 3 rows in every class so stratified "
            "training and evaluation can run."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Target column": item.column,
                        "Classes": item.classes,
                        "Smallest class": item.minimum_class_samples,
                        "Missing values": item.missing_values,
                    }
                    for item in eligible_assessments
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        rejected_assessments = [
            assessment for assessment in assessments if not assessment.eligible
        ]
        if rejected_assessments:
            with st.expander(
                f"Why {len(rejected_assessments)} other columns cannot be targets"
            ):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Column": item.column,
                                "Classes": item.classes,
                                "Smallest class": item.minimum_class_samples,
                                "Missing values": item.missing_values,
                                "Why unavailable": item.reason,
                            }
                            for item in rejected_assessments
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

        default_target = st.session_state.get("remote_target_column")
        target_columns = [item.column for item in eligible_assessments]
        default_target_index = target_columns.index(default_target) if default_target in target_columns else None
        target_column = st.selectbox(
            "Select a target column to evaluate",
            target_columns,
            index=default_target_index,
        )
        model_names = [definition.display_name for definition in MODEL_DEFINITIONS]
        selected_model_names = st.multiselect(
            "Models to train and compare",
            model_names,
            default=model_names,
            help="All five are selected by default. Choose fewer models for a shorter run.",
        )
        st.caption(
            "Clicking Run selected models fits each checked classifier again. "
            "Changing the results-inspection dropdown only reuses the finished run."
        )
        selected_model_definitions = tuple(
            definition
            for definition in MODEL_DEFINITIONS
            if definition.display_name in selected_model_names
        )
        cache_key = (
            st.session_state["remote_dataset_url"],
            target_column,
            tuple(definition.slug for definition in selected_model_definitions),
        )
        run_experiment = st.button("Run selected models", type="primary")
        cached_result = st.session_state.get("generic_evaluation_result")
        show_cached_result = cached_result is not None and cached_result[0] == cache_key

        if not selected_model_definitions:
            st.info("Select at least one model to run an experiment.")
        elif run_experiment or show_cached_result:
            try:
                show_results(
                    remote_data,
                    target_column,
                    reference=False,
                    model_definitions=selected_model_definitions,
                    cache_key=cache_key,
                )
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
