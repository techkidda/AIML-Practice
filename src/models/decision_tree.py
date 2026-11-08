from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel:
    """Factory for the bounded decision-tree classifier."""

    @staticmethod
    def build() -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            random_state=69,
        )
