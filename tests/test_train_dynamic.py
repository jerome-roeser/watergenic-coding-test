import unittest
import pandas as pd
from pathlib import Path
from click.testing import CliRunner

from src.watergenic_coding_test.train_dynamic import main


# Copilot (Claude Sonnet 3.5) was used to generate the test cases below.
# The code was reviewed, tested and modified as necessary.
class TestTrainDynamicInput(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.df = self.df = pd.DataFrame(
            {
                "unnamed_0": [0, 1, 2, 3, 4],
                "time": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],
                "input_variable1": [1, 2, 3, 4, 5],
                "input_variable2": [5, 4, 3, 2, 1],
                "target_variable": [10, 20, 30, 40, 50],
            }
        )
        self.filepath = Path('tests/tmp_df.csv')
        self.df.to_csv(self.filepath)


    def test_n_data_points_lower_than_min_required(self):
        # should raise an error since n_data_points lower than minimum_data_points
        # required for training
        minimum_data_points = 4
        n_data_points = minimum_data_points - 1
        with self.assertRaises(ValueError):
            self.runner.invoke(main, ["--n_data_points", n_data_points,
                                      "--minimum_data_points", minimum_data_points],
                               catch_exceptions=False)

    def test_len_df_lower_than_min_required(self):
        # should raise error since len(df) is lower than minimum_data_points
        # required for training
        length_df = len(self.df)
        minimum_data_points = length_df + 2
        n_data_points = minimum_data_points + 1
        with self.assertRaises(ValueError):
            self.runner.invoke(main, [
                "--input_file", self.filepath,
                "--n_data_points", n_data_points,
                "--minimum_data_points", minimum_data_points],
                catch_exceptions=False
                )

    def test_len_df_lower_than_n_data_point(self):
        # should raise error since n_data_points higher than len(df)
        # i.e. not enough data in the DataFrame
        length_df = len(self.df)
        n_data_points = length_df + 2
        minimum_data_points = length_df - 1
        with self.assertRaises(ValueError):
            self.runner.invoke(main, [
                "--input_file", self.filepath,
                "--n_data_points", n_data_points,
                "--minimum_data_points", minimum_data_points],
                catch_exceptions=False
                )

    def test_valid_n_data_points(self):
        # Test with a valid number of data points
        length_df = len(self.df)
        n_data_points = length_df
        minimum_data_points = length_df - 1
        result = self.runner.invoke(main, [
                "--input_file", self.filepath,
                "--n_data_points", n_data_points,
                "--minimum_data_points", minimum_data_points],
                catch_exceptions=False
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("✅ Model trained successfully.", result.output)
