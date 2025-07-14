import yaml

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

#### CONFIG ####

def load_config(file_path="./config.yaml"):
    """
    Loads the configuration from the YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration file and make it available globally
config = load_config()



#### INPUT VALIDATION ####

# Generated with Copilot.
def validate_input_file(file_path: Path) -> bool:
    """
    Validate if the input file exists and is a CSV or JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the input file.

    Returns
    -------
    bool
        True if the file is valid, False otherwise.
    """
    if not file_path.exists():
        print(f"❗ The file {file_path} does not exist.")
        return False

    if not (file_path.suffix == '.csv'):
        print(f"❗ The file {file_path} is not a valid CSV file.")
        return False

    return True

# Genereated with Copilot.
def validate_input_file_dynamic(file_path: Path) -> bool:
    """
    Validate if the input file exists and is a CSV or JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the input file.

    Returns
    -------
    bool
        True if the file is valid, False otherwise.
    """
    if not file_path.exists():
        print(f"❗ The file {file_path} does not exist.")
        return False

    if not (file_path.suffix == '.csv' or file_path.suffix == '.json'):
        print(f"❗ The file {file_path} is not a valid CSV or JSON file.")
        return False

    return True


# Generated with Copilot.
def validate_input_list_dynamic(input_list: List) -> bool:
    """
    The list should contain 4 lists with the following data points
    The sublists should have the same lengths
    The last 3 sublists should contain only floats

    """
    if not isinstance(input_list, list) or len(input_list) != 4:
        print("❗ Input must be a list of 4 lists.")
        return False

    for sublist in input_list:
        if not isinstance(sublist, list):
            print("❗ Each element of the input list must be a list.")
            return False

    if not all(len(sublist) == len(input_list[0]) for sublist in input_list):
        print("❗ All sublists must have the same length.")
        return False

    if not all(isinstance(item, float) for item in input_list[1:]):
        print("❗ The last 3 sublists must contain float values.")
        return False

    return True

# Generated with Copilot. Checked, edited and validated.
def validate_dataframe_format(df: pd.DataFrame) -> bool:
    """
    Validate if the DataFrame has the correct format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.

    Returns
    -------
    bool
        True if the DataFrame is valid, False otherwise.
    """
    required_columns = ['time', 'input_variable1', 'input_variable2', 'target_variable']

    if not all(col in df.columns for col in required_columns):
        print("❗ DataFrame must contain the columns: time, input_variable1, input_variable2, target_variable.")
        return False

    # Check if input_variable1, input_variable2 and target variable are numeric
    numeric_columns = df.select_dtypes(include=[np.number])
    if not all(col in numeric_columns for col in required_columns[1:]):
        print("❗ input_variable1, input_variable2 and target_variable must be numeric.")
        return False

    return True
