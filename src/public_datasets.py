from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicDataset:
    name: str
    url: str
    rows: int
    features: int
    target_column: str

    @property
    def option_label(self) -> str:
        return f"{self.name} ({self.rows:,} rows, {self.features} features)"


CUSTOM_DATASET_OPTION = "Use a custom dataset URL"

PUBLIC_DATASETS = (
    PublicDataset(
        "Adult Census Income",
        "https://archive.ics.uci.edu/static/public/2/data.csv",
        48_842,
        14,
        "income",
    ),
    PublicDataset(
        "Credit Approval",
        "https://archive.ics.uci.edu/static/public/27/data.csv",
        690,
        15,
        "A16",
    ),
    PublicDataset(
        "Letter Recognition",
        "https://archive.ics.uci.edu/static/public/59/data.csv",
        20_000,
        16,
        "lettr",
    ),
    PublicDataset(
        "Mushroom",
        "https://archive.ics.uci.edu/static/public/73/data.csv",
        8_124,
        22,
        "poisonous",
    ),
    PublicDataset(
        "Optical Recognition of Handwritten Digits",
        "https://archive.ics.uci.edu/static/public/80/data.csv",
        5_620,
        64,
        "class",
    ),
    PublicDataset(
        "Pen-Based Handwritten Digits",
        "https://archive.ics.uci.edu/static/public/81/data.csv",
        10_992,
        16,
        "Class",
    ),
    PublicDataset(
        "Spambase",
        "https://archive.ics.uci.edu/static/public/94/data.csv",
        4_601,
        57,
        "Class",
    ),
    PublicDataset(
        "Bank Marketing",
        "https://archive.ics.uci.edu/static/public/222/data.csv",
        45_211,
        16,
        "y",
    ),
    PublicDataset(
        "Online Shoppers Purchasing Intention",
        "https://archive.ics.uci.edu/static/public/468/data.csv",
        12_330,
        17,
        "Revenue",
    ),
    PublicDataset(
        "Dry Bean",
        "https://archive.ics.uci.edu/static/public/602/data.csv",
        13_611,
        16,
        "Class",
    ),
)

DATASET_BY_LABEL = {
    dataset.option_label: dataset for dataset in PUBLIC_DATASETS
}
