from pickle import load
from typing import Union
import pandas as pd
import click
import mlflow


from pathlib import Path

from sklearn.metrics import mean_absolute_percentage_error, r2_score

from src.watergenic_coding_test.params import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.utils.utils import config, load_data_from_file, validate_pred_dataframe_format

# We'll make predicitons fron the Test set loacted in the data folder
PREDICT_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['test'])

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = config['mlflow']['experiment_name']
MLFLOW_URI = config['mlflow']['uri']
MLFLOW_TRACKING_SERVER = config['mlflow']['tracking_server']



def predict(df, mlflow_tracking_server: bool = False) -> Union[pd.Series, float]:
    """ Predict the target variable using a pre-trained model.

    The function validates the DataFrame format, sets up MLflow for tracking,
    and makes predictions using a pre-trained model.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input data. It must have the columns:
        'input_variable1', and optionally 'target_variable'.
    mlflow_tracking_server : bool, optional
        If True, sets up MLflow to track the model predictions on a remote server.
        If False, uses local MLflow tracking. Default is False.

    Raises
    ------
    ValueError
        If the DataFrame does not have the correct format or if the input data is invalid.

    Returns
    -------
    Union[pd.Series, float]
        A Series containing the predicted values if 'target_variable' is not in the DataFrame,
        or a float representing the R2 score if 'target_variable' is present.

   """

    if not validate_pred_dataframe_format(df):
        raise ValueError("DataFrame does not have the correct format.")

    ###### MLflow Setup ######
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if mlflow_tracking_server:
        print("Setting up MLflow tracking server...")
        print(f"❗ Make sure the MLflow tracking server is running at {MLFLOW_URI} ❗")
        print("You can start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        mlflow.set_tracking_uri(MLFLOW_URI)
    else:
        print("Using local MLflow tracking...")
        print(f"Start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        print(f"🏃 View runs and 🧪 experiments at: {MLFLOW_URI}")

    # Get the model and check if it exists
    with open(Path(LOCAL_MODELS_PATH).joinpath('model.pkl'), 'rb') as model_file:
        pipeline = load(model_file)

    try:
        assert pipeline is not None, "Model pipeline is None. Please check the model file."
    except AssertionError as e:
        raise ValueError("Model pipeline is None. Please check the model file.")

    y_pred = pipeline.predict(df)

    # If the target variable is present, we return the R2 and MAPE scores and log them to MLflow
    if 'target_variable' in df.columns:
       with mlflow.start_run():
        params = {
            "context": "predict",
        }
        mlflow.log_params(params)

        r2 = r2_score(df['target_variable'], y_pred)
        mape = mean_absolute_percentage_error(df['target_variable'], y_pred)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        return y_pred, r2, mape

    return y_pred

@click.command()
@click.option('--input_file', '-i', type=click.Path(exists=True), default=PREDICT_DATA_FILE, help='Path to the input data file (CSV or JSON).')
def main(input_file: Union[Path, str]):
    """
    Main function to make predictions with the model.

    The function loads the prediction data from CLI or config file, validates if
    the input file is a valid CSV or JSON file, and makes predictions using the pre-trained model.

    Parameters
    ----------
    input_file : Union[Path, str]
        Path to the input data file (CSV or JSON). If not provided, defaults to the path defined in the PREDICT_DATA_FILE constant.

    Raises
    ------
    ValueError
        If the input file is not a valid CSV or JSON file, or if the DataFrame
        does not have the correct format.

    Returns
    -------
    None
    """

    train_input = Path(input_file) if input_file else PREDICT_DATA_FILE

    pred_df = load_data_from_file(train_input)

    # Let's precit 10 data points only
    if 'target_variable' in pred_df.columns:
        y_pred, r2, mape = predict(
            pred_df.sample(n=10, replace=True),
            mlflow_tracking_server=MLFLOW_TRACKING_SERVER
            )
        print("\n", "=" * 50)
        print(f"Predictions: {y_pred}")
        print("\n", "=" * 50)
        print("The mean absolute percentage error (MAPE) prediction error")
        print(f"=> {mape:.5g} % (R2 =  {r2:.5g})\n")
    else:
        y_pred = predict(
            pred_df.sample(n=10, replace=True),
            mlflow_tracking_server=MLFLOW_TRACKING_SERVER
            )
        print("\n", "=" * 50)
        print(f"Predictions: {y_pred}")
if __name__ == "__main__":
    main()
