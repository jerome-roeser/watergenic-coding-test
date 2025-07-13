from pickle import load
import pandas as pd
import click


from pathlib import Path

from src.watergenic_coding_test.params import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.utils.utils import config

# We'll make predicitons fron the Test set loacted in the data folder
PREDICT_DATA_FILE = Path(LOCAL_DATA_PATH).joinpath(config['files']['test'])

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


def predict(df):
    with open(Path(LOCAL_MODELS_PATH).joinpath('model.pkl'), 'rb') as model_file:
        pipeline = load(model_file)
    y_pred = pipeline.predict(df)

    # add score

    return y_pred

@click.command()
@click.option('--input_data', '-i', type=click.Path(exists=True), default=PREDICT_DATA_FILE, help='Path to the input data file (CSV or JSON).')
def main():
    pred_df = load_data(PREDICT_DATA_FILE)
    y_pred = predict(pred_df)
    return y_pred

if __name__ == "__main__":
    main()
