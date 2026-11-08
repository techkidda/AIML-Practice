from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


class SupportVectorMachineModel:
    """Factory for a calibrated RBF-kernel support vector classifier."""

    @staticmethod
    def build() -> CalibratedClassifierCV:
        return CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=69),
            method="sigmoid",
            cv=2,
            ensemble=False,
        )
