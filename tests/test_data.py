import pytest
import pandas as pd
import numpy as np
from src.pipelines import get_preprocessing_pipeline


def test_pipeline_transforms_correctly():
    # Setup mock dataframe matching UCI scheme
    mock_data = pd.DataFrame({
        "age": [55, 60], "sex": [1, 0], "cp": [3, 2], "trestbps": [140, 120],
        "chol": [250, 240], "fbs": [0, 1], "restecg": [0, 1], "thalach": [150, 130],
        "exang": [0, 1], "oldpeak": [2.3, 1.5], "slope": [0, 1], "ca": [0.0, 2.0],
        "thal": [2.0, 3.0]
    })

    preprocessor = get_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(mock_data)

    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()