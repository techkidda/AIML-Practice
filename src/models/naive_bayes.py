from sklearn.naive_bayes import GaussianNB


class NaiveBayesModel:
    """Factory for the Gaussian Naive Bayes classifier."""

    @staticmethod
    def build() -> GaussianNB:
        return GaussianNB()
