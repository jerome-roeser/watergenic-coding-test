"""
Task 2: Dynamic ML Model
    ●​ Refactor your code to accept a dynamic number of data points
        ○​ Must reject fewer than 4 data points with a clear error. You can use subset from 202502_data_train.csv
        ○​ Should accept .csv, .json, or list formats
        ○​ Must still log metrics using MLflow
"""


from pathlib import Path
from typing import Union, List
import json

import pandas as pd

from src.watergenic_coding_test.train import train_model
from src.watergenic_coding_test.params import LOCAL_DATA_PATH
from src.utils.utils import config

TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['train'])


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.
    """
    print (f"Loading data from {file_path.stem}...")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    print("✅ Data loaded successfully.")
    if file_path.suffix == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .json file.")

    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def main(
    input_data: Union[Path, List[List]] = TRAIN_DATA_FILE,
    n_data_points: int = 5
    ) -> None:

    # Define / hardcode the column names in case of json or list input
    columns = ['time', 'input_variable1', 'input_variable2', 'target_variable']

    # check if the input is a valid CSV file, a valid .json file or a List of data points
    if isinstance(input_data, List):
        # We assume the List input is a list of 4 lists containing the time,
        # input_variable1, input_variable2, and target_variable data points
        train_df = pd.DataFrame(input_data).T
        train_df.columns = columns
    elif isinstance(input_data, Path):
        if input_data.suffix == '.csv' or input_data.suffix == '.json':
            train_df = load_data(input_data)
        else:
            raise ValueError("Input data must be a Path to a CSV or JSON file.")
        train_df = load_data(input_data)
    else:
        raise ValueError("Input data must be a Path to a CSV file, a Path to a JSON File or a List of data points.")


    # train the model with a sample of 5 data points
    train_df = train_df.sample(n_data_points, replace=True)

    if len(train_df) < 4:
        raise ValueError("You must provide at least 4 data points for training.")


    train_score = train_model(
        train_df,
        mlflow_tracking_server=config['mlflow']['tracking_server']
        )
    print(f"Training score: {train_score:.2f}")


if __name__ == "__main__":
    n_data_points = config['n_data_points']
    main(n_data_points)
