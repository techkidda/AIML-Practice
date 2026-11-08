import unittest
from urllib.parse import urlparse

from src.public_datasets import (
    CUSTOM_DATASET_OPTION,
    DATASET_BY_LABEL,
    PUBLIC_DATASETS,
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


if __name__ == "__main__":
    unittest.main()
