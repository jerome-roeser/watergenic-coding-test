from pathlib import Path
from typing import Union, List

import click
import pandas as pd

from src.watergenic_coding_test.train import train_model
from src.watergenic_coding_test.params import LOCAL_DATA_PATH
from src.utils.utils import (
    config,
    validate_input_file_dynamic,
    validate_input_list_dynamic,
    load_data_from_file,
)


TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config["files"]["train"])
MLFLOW_TRACKING_SERVER = config["mlflow"]["tracking_server"]
N_DATA_POINTS = config["n_data_points"]


@click.command()


@click.option(
    "--input_file",
    "-i",
    type=click.Path(exists=True),
    default=TRAIN_DATA_FILE,
    help="Path to the input data file (CSV or JSON).",
)
@click.option(
    "--n_data_points",
    "-n",
    type=int,
    help="Number of data points to sample for training.",
)

def main(
    input_file: Union[Path, List[List]] = TRAIN_DATA_FILE,
    n_data_points: int = 5,
) -> None:
    """
    Main function to train the model with dynamic input.

    The function loads the training data from CLI or config file, validates if
    the input file is a valid CSV or JSON file, and trains the model with a dynamic
    number of data points.

    Parameters
    ----------
    input_file : Union[Path, List[List]]
        Path to the input data file (CSV or JSON) or a List of data points.
        If not provided, defaults to the path defined in the TRAIN_DATA_FILE constant.
    n_data_points : int
        Number of data points to sample for training. If not provided, defaults to the value defined
        in the N_DATA_POINTS constant.

    Raises
    ------
    ValueError
        If the input file is not a valid CSV or JSON file, or if the DataFrame does not have the correct format,
        or if the number of data points is less than 4.

    Returns
    -------
    None
    """

    train_input = Path(input_file) if input_file else TRAIN_DATA_FILE
    n_train = n_data_points if n_data_points else N_DATA_POINTS

    # Define / hardcode the column names in case of json or list input
    columns = ["time", "input_variable1", "input_variable2", "target_variable"]

    # check if the input is a valid CSV file, a valid .json file or a List of data points
    if isinstance(train_input, List):
        if validate_input_list_dynamic(train_input):
            train_df = pd.DataFrame(train_input).T
            train_df.columns = columns
        else:
            raise ValueError(
                "Input data as List must follow the format: [[time], [input_variable1], [input_variable2], [target_variable]]."
            )
    elif isinstance(train_input, Path):
        if validate_input_file_dynamic(train_input):
            train_df = load_data_from_file(train_input)
        else:
            raise ValueError("Input data must be a Path to a CSV or JSON file.")
    else:
        raise ValueError(
            "Input data must be a Path to a CSV file, a Path to a JSON File or a List of data points."
        )

    # ensure that the model is trained with at least 4 data points
    if len(train_df) < 4 or n_train < 4:
        raise ValueError("You must provide at least 4 data points for training.")
    # ensure the required number of data points is not too high
    if len(train_df) < n_train:
        raise ValueError(
            f"Not enough data points in the DataFrame. Expected at least {n_train} data points, but got {len(train_df)}."
        )

    # train the model with a dynamic number of data points
    train_df = train_df.sample(n_train, replace=True)

    pipeline, r2, mape = train_model(
        train_df, mlflow_tracking_server=MLFLOW_TRACKING_SERVER
    )
    print("\n", "=" * 50)
    print("The mean absolute percentage error (MAPE) training error:")
    print(
        f"Linear Regression on {n_train} data points: {mape:.5g} % (R2 =  {r2:.5g})\n"
    )


if __name__ == "__main__":
    main()
