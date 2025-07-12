import pandas as pd
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.watergenic_coding_test.params import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.utils.utils import config


# Define the path to the training data file in a robust way
TRAIN_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['train'])


def load_data(file_path: str) -> pd.DataFrame:
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

    return pd.read_csv(file_path)


def train_model(df: pd.DataFrame):
    X_train = df.copy()
    y_train = X_train.pop('target_variable')

    reg = LinearRegression()
    scaler = MinMaxScaler()

    pipeline = Pipeline([
        ('scaler', scaler),
        ('Linear Regressor', reg)]
                        )

    print("Training model...")
    pipeline.fit(df[['input_variable1']], y_train)
    print("✅ Model trained successfully.")


    print("Saving model...")
    with open(Path().joinpath(LOCAL_MODELS_PATH, 'model.pkl'), 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("✅ Model saved successfully.")

def main():
    train_df = load_data(TRAIN_DATA_FILE)
    train_model(train_df)


if __name__ == "__main__":
    main()
