from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Factory for the logistic-regression baseline."""

    @staticmethod
    def build() -> LogisticRegression:
        return LogisticRegression(max_iter=1500, random_state=69)
