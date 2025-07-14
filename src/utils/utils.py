import yaml

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

#### CONFIG ####

def load_config(file_path="./config.yaml"):
    """
    Loads the configuration from the YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file. Default is './config.yaml'.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration file and make it available globally
config = load_config()


##### LOAD DATA #####

def load_data_from_file(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV or JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the CSV/JSON file.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.
    """
    print (f"Loading data from {file_path.stem}...")


    if file_path.suffix == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .json file.")

    print("✅ Data loaded successfully.")
    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    return df


#### INPUT VALIDATION ####

# Copilot (Claude Sonnet 3.5) was used to generate the data validation cases below.
# The code was reviewed, tested and modified as necessary.

def validate_input_file(file_path: Path) -> Union[ValueError, True]:
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
        raise ValueError(f"❗ The file {file_path} does not exist.")

    if not (file_path.suffix == '.csv'):
        raise ValueError(f"❗ The file {file_path} is not a valid CSV file.")

    return True


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
        raise ValueError(f"❗ The file {file_path} does not exist.")


    if not (file_path.suffix == '.csv' or file_path.suffix == '.json'):
        raise ValueError(f"❗ The file {file_path} is not a valid CSV or JSON file.")

    return True



def validate_input_list_dynamic(input_list: List) -> bool:
    """
    The list should contain 4 lists with the following data points
    The sublists should have the same lengths
    The last 3 sublists should contain only floats

    """
    if not isinstance(input_list, list) or len(input_list) != 4:
        raise ValueError("❗ Input must be a list of 4 lists.")

    for sublist in input_list:
        if not isinstance(sublist, list):
            raise ValueError("❗ Each element of the input list must be a list.")

    if not all(len(sublist) == len(input_list[0]) for sublist in input_list):
        raise ValueError("❗ All sublists must have the same length.")

    if not all(isinstance(item, float) for item in input_list[1:]):
        raise ValueError("❗ The last 3 sublists must contain float values.")

    return True


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
        raise ValueError("❗ DataFrame must contain the columns: time, input_variable1, input_variable2, target_variable.")

    # Check if input_variable1, input_variable2 and target variable are numeric
    numeric_columns = df.select_dtypes(include=[np.number])
    if not all(col in numeric_columns for col in required_columns[1:]):
        raise ValueError("❗ input_variable1, input_variable2 and target_variable must be numeric.")

    return True

def validate_pred_dataframe_format(df: pd.DataFrame) -> bool:
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
    if 'target_variable' in df.columns:
        required_columns = ['input_variable1', 'target_variable']
    else:
        required_columns = ['input_variable1']

    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"❗ DataFrame must contain the column: {required_columns}")

    # Check if input_variable1, input_variable2 and target variable are numeric
    numeric_columns = df.select_dtypes(include=[np.number])
    if not all(col in numeric_columns for col in required_columns):
        raise ValueError(f"❗ {required_columns} must be numeric.")

    return True
