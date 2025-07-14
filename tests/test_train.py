import unittest
import pandas as pd

from src.watergenic_coding_test.train import train_model


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        # Setup code to create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'unnamed_0': [0, 1, 2, 3, 4],
            'time': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'input_variable1': [1, 2, 3, 4, 5],
            'input_variable2': [5, 4, 3, 2, 1],
            'target_variable': [10, 20, 30, 40, 50]
        })

    def test_train_model_score(self):
        # Test the train_model function with the sample DataFrame
        try:
            model, score = train_model(self.df, mlflow_tracking_server=False)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)  # score should be positive
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

    def test_train_model_with_invalid_data(self):
        # Test with an invalid DataFrame (missing target variable)
        invalid_df = self.df.drop(columns=['target_variable'])
        with self.assertRaises(ValueError):
            train_model(invalid_df, mlflow_tracking_server=False)

    def test_train_model_with_empty_dataframe(self):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            train_model(empty_df, mlflow_tracking_server=False)

    def test_model_is_not_none(self):
        # Test that the model is not None after training
        model, score = train_model(self.df, mlflow_tracking_server=False)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, type(train_model(self.df, mlflow_tracking_server=False)[0]))
