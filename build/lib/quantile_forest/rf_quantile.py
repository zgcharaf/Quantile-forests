# quantile_forest/rf_quantile.py

import numpy as np
from .utils import rf_gap_quantile_regression_parallel, qrf_predict

class QuantileRegressor:
    def __init__(self, base_regressor, **regressor_params):
        """
        Initialize the QuantileRegressor with a tree-based model.
        
        Parameters:
        - base_regressor: A tree-based regressor class (e.g., RandomForestRegressor, DecisionTreeRegressor).
        - regressor_params: Parameters for the regressor (e.g., n_estimators, max_depth).
        """
        self.model = base_regressor(**regressor_params)
    
    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        """
        self.model.fit(X_train, y_train)
    
    def predict_quantiles(self, X_train, y_train, X_test, quantiles, method='rf_gap'):
        """
        Predict quantiles using either 'rf_gap' or 'qrf' method.
        
        Parameters:
        - X_train: Training features.
        - y_train: Training target.
        - X_test: Test features.
        - quantiles: List of quantiles to predict (e.g., [5, 50, 95]).
        - method: Method to use ('rf_gap' or 'qrf').
        """
        if method == 'rf_gap':
            return rf_gap_quantile_regression_parallel(self.model, X_train, y_train, X_test, quantiles)
        elif method == 'qrf':
            return qrf_predict(self.model, X_test, quantiles)
        else:
            raise ValueError("Method must be either 'rf_gap' or 'qrf'.")

    def evaluate(self, y_test, predictions, quantiles, metric):
        """
        Evaluate predictions using a custom metric.
        
        Parameters:
        - y_test: True values of the test set.
        - predictions: Predicted quantiles.
        - quantiles: List of quantiles to evaluate.
        - metric: A callable function that takes (y_true, y_pred) and returns a score.
        
        Returns:
        A dictionary with quantiles as keys and evaluation scores as values.
        """
        scores = [metric(y_test, predictions[:, i]) for i in range(len(quantiles))]
        return dict(zip(quantiles, scores))
