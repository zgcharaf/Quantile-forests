import pytest
import numpy as np
from quantile_forest.utils import weighted_percentile

def test_weighted_percentile():
    data = np.array([1, 2, 3, 4, 5])
    weights = np.array([1, 1, 1, 1, 1])
    quantiles = [0, 50, 100]
    
    result = weighted_percentile(data, quantiles, sample_weight=weights)
    
    assert np.allclose(result, [1, 3, 5])
