from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefinition:
    slug: str
    display_name: str
    module: str
    class_name: str


MODEL_DEFINITIONS = (
    ModelDefinition(
        "logistic_regression",
        "Logistic Regression",
        "src.models.logistic_regression",
        "LogisticRegressionModel",
    ),
    ModelDefinition(
        "decision_tree",
        "Decision Tree",
        "src.models.decision_tree",
        "DecisionTreeModel",
    ),
    ModelDefinition("knn", "K-Nearest Neighbors", "src.models.knn", "KNNModel"),
    ModelDefinition(
        "naive_bayes",
        "Gaussian Naive Bayes",
        "src.models.naive_bayes",
        "NaiveBayesModel",
    ),
    ModelDefinition(
        "support_vector_machine",
        "Support Vector Machine",
        "src.models.support_vector_machine",
        "SupportVectorMachineModel",
    ),
)

MODEL_BY_SLUG = {definition.slug: definition for definition in MODEL_DEFINITIONS}
