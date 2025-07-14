from pathlib import Path
from typing import Union, List
import json

import click
import pandas as pd

from src.watergenic_coding_test.train import train_model
from src.watergenic_coding_test.params import LOCAL_DATA_PATH
from src.utils.utils import config, validate_input_file_dynamic, validate_input_list_dynamic, validate_dataframe_format


TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['train'])
MLFLOW_TRACKING_SERVER = config['mlflow']['tracking_server']
N_DATA_POINTS = config['n_data_points']


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

    if not validate_dataframe_format(df):
        raise ValueError("DataFrame does not have the correct format.")
    return df

@click.command()
@click.option('--input_file', '-i', type=click.Path(exists=True), default=TRAIN_DATA_FILE, help='Path to the input data file (CSV or JSON).')
@click.option('--n_data_points', '-n', type=int, default=5, help='Number of data points to sample for training.')
def main(
    input_file: Union[Path, List[List]] = TRAIN_DATA_FILE,
    n_data_points: int = 5,
    ) -> None:


    train_input = Path(input_file) if input_file else TRAIN_DATA_FILE
    n_train = n_data_points if n_data_points else N_DATA_POINTS

    # Define / hardcode the column names in case of json or list input
    columns = ['time', 'input_variable1', 'input_variable2', 'target_variable']

    # check if the input is a valid CSV file, a valid .json file or a List of data points
    if isinstance(train_input, List):
        if validate_input_list_dynamic(train_input):
            train_df = pd.DataFrame(train_input).T
            train_df.columns = columns
        else:
            raise ValueError("Input data as List must follow the format: [[time], [input_variable1], [input_variable2], [target_variable]].")
    elif isinstance(train_input, Path):
        if validate_input_file_dynamic(train_input):
            train_df = load_data_from_file(train_input)
        else:
            raise ValueError("Input data must be a Path to a CSV or JSON file.")
    else:
        raise ValueError("Input data must be a Path to a CSV file, a Path to a JSON File or a List of data points.")



    # ensure that the model is trained with at least 4 data points
    if len(train_df) < 4 or n_train < 4:
        raise ValueError("You must provide at least 4 data points for training.")
    # ensure the required number of data points is not too high
    if len(train_df) < n_train:
        raise ValueError(f"Not enough data points in the DataFrame. Expected at least {n_train} data points, but got {len(train_df)}.")

    # train the model with a dynamic number of data points
    train_df = train_df.sample(n_train, replace=True)

    train_score = train_model(
        train_df,
        mlflow_tracking_server= MLFLOW_TRACKING_SERVER
        )
    print(f"Training score: {train_score:.2f}")


if __name__ == "__main__":
    main()
