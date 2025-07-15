from pathlib import Path
from src.utils.utils import config


repo_path = Path(__file__).parent.parent.parent

LOCAL_DATA_PATH = Path(repo_path).joinpath(config["folders"]["data"])
LOCAL_MODELS_PATH = Path(repo_path).joinpath(config["folders"]["models"])
