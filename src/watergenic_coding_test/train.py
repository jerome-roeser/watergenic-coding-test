import pandas as pd
import pickle
from pathlib import Path
from typing import Union, Tuple

import click
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from src.watergenic_coding_test.params import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.utils.utils import config, load_data_from_file, validate_input_file, validate_dataframe_format


# Define the path to the training data file in a robust way
TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['train'])

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = config['mlflow']['experiment_name']
MLFLOW_URI = config['mlflow']['uri']
MLFLOW_TRACKING_SERVER = config['mlflow']['tracking_server']


def train_model(
    df: pd.DataFrame,
    mlflow_tracking_server: bool = False
    )-> Tuple[Pipeline, float, float]:
    """ Train a machine learning model using the provided DataFrame.

    The function validates the DataFrame format, sets up MLflow for tracking,
    and trains a Linear Regression model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training data. It must have the columns:
        'time', 'input_variable1', 'input_variable2', and 'target_variable'.
    mlflow_tracking_server : bool, optional
        If True, sets up MLflow to track the model training on a remote server.
        If False, uses local MLflow tracking. Default is False.

    Raises
    ------
    ValueError
        If the DataFrame does not have the correct format or if the input data is invalid.

    Returns
    -------
    Tuple[Pipeline, float, float]
        A tuple containing the trained model pipeline and the training score.
        The pipeline is a scikit-learn Pipeline object, and the r2 and mape score as floats representing
        the model's performance on the training data.
    """

    if not validate_dataframe_format(df):
        raise ValueError("DataFrame does not have the correct format.")

    ###### MLflow Setup ######
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if mlflow_tracking_server:
        print("\nSetting up MLflow tracking server...")
        print(f"❗ Make sure the MLflow tracking server is running at {MLFLOW_URI} ❗")
        print("You can start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        mlflow.set_tracking_uri(MLFLOW_URI)
    else:
        print("\nUsing local MLflow tracking...")
        print(f"Start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        print(f"🏃 View runs and 🧪 experiments at: {MLFLOW_URI}")

    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()


    ###### Model Training Pipeline ######
    X_train = df.copy()
    y_train = X_train.pop('target_variable')

    num_transformer = make_pipeline(
        KNNImputer(),
        MinMaxScaler()
        )

    num_prepocessor = make_column_transformer(
        (num_transformer, ["input_variable1"]),
        remainder="drop"
        )

    pipeline = make_pipeline(
        num_prepocessor,
        LinearRegression()
    )

    ###### Training and saving the Model ######
    with mlflow.start_run():
        params = {
            "context": "train",
            "training_set_size": X_train.shape[0],
        }
        mlflow.log_params(params)
        print(f"\nTraining model on {X_train.shape[0]} data points...")
        pipeline.fit(X_train, y_train)
        print("✅ Model trained successfully.")

        y_pred = pipeline.predict(X_train)
        r2 = r2_score(df['target_variable'], y_pred)
        mape = mean_absolute_percentage_error(df['target_variable'], y_pred)


    print("Saving model...")
    if not Path(LOCAL_MODELS_PATH).exists():
        Path(LOCAL_MODELS_PATH).mkdir(parents=False, exist_ok=True)

    with open(Path().joinpath(LOCAL_MODELS_PATH, 'model.pkl'), 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("✅ Model saved successfully.")
    return pipeline, r2, mape


@click.command()
@click.option('--input_file', '-i',
              type=click.Path(),
              default=TRAIN_DATA_FILE,
              help='Path to the input data file (CSV or JSON).'
              )
def main(input_file: Union[Path, str]):
    """
    Main function to train the model.

    The function loads the training data from CLI or config file, validates if
    the input file is a valid CSV or JSON file, and trains the model.

    Parameters
    ----------
    input_file : Union[Path, str]
        Path to the input data file (CSV or JSON). If not provided, defaults to the
        path defined in the TRAIN_DATA_FILE constant.

    Raises
    ------
    ValueError
        If the input file is not a valid CSV or JSON file, or if the DataFrame does not have the correct format.

    Returns
    -------
    None
    """

    train_input = Path(input_file) if input_file else TRAIN_DATA_FILE

    if validate_input_file(train_input):
        train_df = load_data_from_file(train_input)


    # train the model with a sample of 5 data points
    train_df = train_df.sample(5, replace=True)
    pipeline, r2, mape = train_model(
        train_df,
        mlflow_tracking_server=MLFLOW_TRACKING_SERVER
        )
    print("\n", "=" * 50)
    print("The mean absolute percentage error (MAPE) training error:")
    print(f"Linear Regression on 5 data points: {mape:.5g} % (R2 =  {r2:.5g})\n")



if __name__ == "__main__":
    main()
