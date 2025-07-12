import yaml


def load_config(file_path="./config.yaml"):
    """
    Loads the configuration from the YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration file and make it available globally
config = load_config()
