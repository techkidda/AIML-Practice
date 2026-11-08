from sklearn.neighbors import KNeighborsClassifier


class KNNModel:
    """Factory for the distance-weighted K-nearest-neighbors classifier."""

    @staticmethod
    def build() -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
