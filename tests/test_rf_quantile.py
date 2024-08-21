import pytest
from quantile_forest.rf_quantile import QuantileForest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_quantile_forest_fit_predict():
    data = fetch_california_housing()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, data.target, test_size=0.2, random_state=42)

    qf = QuantileForest(n_estimators=10, max_depth=5, random_state=42)
    qf.fit(X_train, y_train)
    
    quantiles = [5, 50, 95]
    predictions = qf.predict_quantiles(X_test, quantiles, method='rf_gap')
    
    assert predictions.shape == (X_test.shape[0], len(quantiles))

def test_quantile_forest_evaluate():
    data = fetch_california_housing()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, data.target, test_size=0.2, random_state=42)

    qf = QuantileForest(n_estimators=10, max_depth=5, random_state=42)
    qf.fit(X_train, y_train)
    
    quantiles = [5, 50, 95]
    predictions = qf.predict_quantiles(X_test, quantiles, method='rf_gap')
    
    evaluation = qf.evaluate(y_test, predictions, quantiles)
    
    assert len(evaluation) == len(quantiles)
