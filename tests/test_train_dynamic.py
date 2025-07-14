import unittest
from click.testing import CliRunner

from src.watergenic_coding_test.train_dynamic import main


# Copilot (Claude Sonnet 3.5) was used to generate the test cases below.
# The code was reviewed, tested and modified as necessary.
class TestTrainDynamicInput(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_too_few_data_points(self):
        # Test with an invalid CSV file
        with self.assertRaises(ValueError):
            self.runner.invoke(main, ['--n_data_points', 3],
                          catch_exceptions=False)

    def test_too_many_data_points(self):
        # Test with an invalid CSV file
        with self.assertRaises(ValueError):
            self.runner.invoke(main, ['--n_data_points', 10_000],
                          catch_exceptions=False)

    def test_valid_n_data_points(self):
        # Test with a valid number of data points
        result = self.runner.invoke(main, ['--n_data_points', 8])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("✅ Model trained successfully.", result.output)
