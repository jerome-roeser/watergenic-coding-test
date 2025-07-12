import pandas as pd
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

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

    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    return df


def train_model(df: pd.DataFrame):
    X_train = df.copy()
    y_train = X_train.pop('target_variable')

    num_transformer = make_pipeline(
        KNNImputer(),
        MinMaxScaler()
        )

    num_prepocessor = make_column_transformer(
        (num_transformer, ["input_variable1", "input_variable2"]),
        remainder="drop"
        )

    pipeline = make_pipeline(
        num_prepocessor,
        LinearRegression()
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("✅ Model trained successfully.")


    print("Saving model...")
    with open(Path().joinpath(LOCAL_MODELS_PATH, 'model.pkl'), 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("✅ Model saved successfully.")

def main():
    train_df = load_data(TRAIN_DATA_FILE)
    train_model(train_df.sample(5, replace=True))


if __name__ == "__main__":
    main()
