import unittest
import numpy as np
import pandas as pd
from pickle import load
from pathlib import Path
from typing import Tuple
from click.testing import CliRunner

from src.watergenic_coding_test.params import LOCAL_MODELS_PATH
from src.watergenic_coding_test.predict import main, predict

# Copilot (Claude Sonnet 3.5) was used to generate the test cases below.
# The code was reviewed, tested and modified as necessary.
class TestPredictValues(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

        self.pred_df = pd.DataFrame({
            'input_variable1': [1, 2, 3, 4, 5],
            'input_variable2': [5, 4, 3, 2, 1],
        })

        self.test_df = pd.DataFrame({
            'input_variable1': [1, 2, 3, 4, 5],
            'input_variable2': [5, 4, 3, 2, 1],
            'target_variable': [10, 20, 30, 40, 50]
        })

        with open(Path(LOCAL_MODELS_PATH).joinpath('model.pkl'), 'rb') as model_file:
            pipeline = load(model_file)
        self.model = pipeline

    def test_predict_values(self):
        # Test the predict function with the sample DataFrame
        predictions = predict(self.model, self.pred_df, mlflow_tracking_server=False)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.pred_df))

    def test_test_model(self):
        # TWhen a target variable is present, the predict function serves as
        # a test for the model, returning the R2 and MAPE scores.
        predictions = predict(self.model, self.test_df, mlflow_tracking_server=False)
        self.assertIsInstance(predictions, Tuple)
        y_pred, r2, mape = predictions
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mape, float)
        self.assertEqual(len(y_pred), len(self.test_df))

    def test_invalid_dataframe(self):
        # Test with an invalid DataFrame (missing input variables)
        invalid_df = pd.DataFrame({
            'input_variable1': ['1', 2, 3, 4, 5],
            'input_variable2': [5, 4, 3, 2, 1],
        })
        with self.assertRaises(ValueError):
            predict(self.model, invalid_df, mlflow_tracking_server=False)

    def test_raise_error_when_model_is_None(self):
        model = None
        with self.assertRaises(ValueError):
            predict(model, self.pred_df, mlflow_tracking_server=False)
