import unittest
from urllib.parse import urlparse

import pandas as pd

from src.public_datasets import (
    CUSTOM_DATASET_OPTION,
    DATASET_BY_LABEL,
    PUBLIC_DATASETS,
    normalize_public_dataset,
)


class PublicDatasetCatalogTests(unittest.TestCase):
    def test_catalog_contains_ten_distinct_qualifying_csv_urls(self):
        self.assertEqual(len(PUBLIC_DATASETS), 10)
        self.assertEqual(len(DATASET_BY_LABEL), 10)
        self.assertNotIn(CUSTOM_DATASET_OPTION, DATASET_BY_LABEL)

        urls = set()
        for dataset in PUBLIC_DATASETS:
            with self.subTest(dataset=dataset.name):
                parsed_url = urlparse(dataset.url)
                self.assertEqual(parsed_url.scheme, "https")
                self.assertEqual(parsed_url.hostname, "archive.ics.uci.edu")
                self.assertTrue(parsed_url.path.endswith("/data.csv"))
                self.assertGreaterEqual(dataset.rows, 600)
                self.assertGreaterEqual(dataset.features, 12)
                self.assertTrue(dataset.target_column)
                self.assertNotIn(dataset.url, urls)
                urls.add(dataset.url)

    def test_adult_target_normalizes_train_test_punctuation_to_two_classes(self):
        adult = next(item for item in PUBLIC_DATASETS if item.name == "Adult Census Income")
        dataset = pd.DataFrame(
            {"income": [" <=50K", " >50K", " <=50K.", " >50K."]}
        )

        normalized = normalize_public_dataset(dataset, adult.url)

        self.assertEqual(normalized["income"].tolist(), ["<=50K", ">50K", "<=50K", ">50K"])
        self.assertEqual(normalized["income"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
