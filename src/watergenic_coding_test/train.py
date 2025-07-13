import pandas as pd
import pickle
from pathlib import Path

import click
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from src.watergenic_coding_test.params import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.utils.utils import config


# Define the path to the training data file in a robust way
TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['train'])


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    file_path : Path
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

    df = pd.read_csv(file_path)

    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def train_model(df: pd.DataFrame, mlflow_tracking_server: bool = False) -> float:

    ###### MLflow Setup ######
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    if mlflow_tracking_server:
        print("Setting up MLflow tracking server...")
        print(f"❗ Make sure the MLflow tracking server is running at {config['mlflow']['uri']} ❗")
        print("You can start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        mlflow.set_tracking_uri(config['mlflow']['uri'])
    else:
        print("Using local MLflow tracking...")
        print(f"Start a local MLflow server with UI by running the command **mlflow ui** in your terminal")
        print(f"🏃 View runs and 🧪 experiments at: {config['mlflow']['uri']}")

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
        print(f"Training model on {X_train.shape[0]} samples with \
{X_train.shape[1]} features...")
        pipeline.fit(X_train, y_train)
        print("✅ Model trained successfully.")
        score = pipeline.score(X_train, y_train)


    print("Saving model...")
    try:
        Path(LOCAL_MODELS_PATH).mkdir(parents=False, exist_ok=True)
    except Exception as e:
        print(f"❗ Error creating model directory: {e}")
        raise

    with open(Path().joinpath(LOCAL_MODELS_PATH, 'model_24.pkl'), 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("✅ Model saved successfully.")
    return score


@click.command()
@click.option('--input_data', type=click.Path(exists=True), default=TRAIN_DATA_FILE, help='Path to the input data file (CSV or JSON).')
def main():
    train_df = load_data(TRAIN_DATA_FILE)

    # train the model with a sample of 5 data points
    train_df = train_df.sample(5, replace=True)
    train_score = train_model(
        train_df,
        mlflow_tracking_server=config['mlflow']['tracking_server']
        )
    print(f"Training score: {train_score:.2f}")


if __name__ == "__main__":
    main()
