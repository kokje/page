import pandas as pd
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
import numpy as np

filterwarnings('ignore')


class ModelEvaluation:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.features = ['age','bmi','PC1','PC2','PC3','PC4','PC5']
        self.label = ['t2d']

    def test(self, weights, intercepts):
        X_test = pd.DataFrame(self.df, columns=self.features)
        y_test = pd.DataFrame(self.df, columns=self.label)
        y_test = y_test.values.ravel()

        lr = LogisticRegression()
        lr.fit(X_test, y_test)
        lr.coef_ = weights  # override weights and coefficients
        lr.intercept_ = intercepts
        return lr.score(X_test, y_test)

    def converged(self, personal, federated, tolerance):
        if not federated:
            return False
        personal_weights, personal_intercepts = personal
        federated_weights, federated_intercepts = federated

        weights_differences = np.abs(federated_weights - personal_weights)
        intercepts_differences = np.abs(federated_intercepts - personal_intercepts)
        return (weights_differences < tolerance).all() and (
                intercepts_differences < tolerance).all()
