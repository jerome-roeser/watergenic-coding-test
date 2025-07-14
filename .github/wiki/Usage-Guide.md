The current package aims at predicting a `target_variable` from a set of 2 input variables with a very simple Machine Learning algorithm.

# 1. General Usage

## 1.1 Training the model
The package allows you to train a model based of labeled data. There are 2 implemented ways to train the model:
* A standard training procedure:
    * based on **5 data points** sampled from a local CSV file
    * the model is saved locally
    * the training parameters and training performance ar elogged with `mlflow`
    * the script prints the training metrics

* A dynamic procedure:
    * the number of **data points** used to train the model can be **chosen dynamically** (but must be equal to or higher than 4)
    * the input format can be CSV, JSON files or in List format.
    * the model is saved locally
    * the training parameters and training performance ar elogged with `mlflow`

## 1.2 Testing the model and predicting values
The Model cna be tested for its performance on unseen data and be served to predict
the target values with a single script.
The predict script can be used to:
* **test the model**: if a CSV file **containing input variables & labeled data** is provided
    * in that case the performance metrics are automatically logged with `mlflow`
    * the script prints the predicted values
    * the script prints the testing metrics
* **predict values**: if a CSV file containing **only the input variables** is provided
    * int that case the scripts only prints the predicted values

# 2. Running the script
There are 2 ways for the user to run the script

## 2.1 Working with the `config.yaml` file and `make` commands

For this the `config.yaml` should be properly edited:
* files
    * `train`: the name of the filen used for training the algorithm (should be a CSV file for the basic training script can be CSV or JSON for the dynamic training script)
    * `test`: the name of the file used for predictions
* folders:
    * `data`: the relative path to the data folder containng the train/test files
    * `models`: the relative path of the folder used to store the trained model
* mlflow:
    * `experiment_name`: the name for the MLFlow experiments
    * `tracking_server`: a boolean to activate MLFlow tracking server (if True, make sure tracking server is running)
    * `uri`: the URI of the tracking server

* `n_data_points`: Number of data points to use for dynamic training (must be >= 4)

The training and predicition scripts can be run with basic `make` commands:
``` bash
make train
make train-dynamic
make predict
```

## 2.2 Working with the CLI
Additionnally the script can be run and the arguments can be called from the
command line interface.

* arguments are optional: if *no arguments are provided* the settings are gathered from the `config.yaml` file
* argument options:
``` text
For Standard Training
Options:
  -i, --input_file PATH        Path to the input data file (CSV).

-----------------------------------------------------------

For Dynamic Training
Options:
  -i, --input_file PATH        Path to the input data file (CSV or JSON).
  -n, --n_data_points INTEGER  Number of data points to sample for training.

-----------------------------------------------------------

For Testing / Predicting
Options:
  -i, --input_file PATH        Path to the input data file (CSV or JSON).
```

Example:

``` bash
python src/watergenic_coding_test/train.py -i data/202502_data_train.csv
python src/watergenic_coding_test/train_dynamic.py -i data/202502_data_train.json -n 8

python src/watergenic_coding_test/predict.py -i data/202302_data_test.csv
```
# 3. Checking model performance with the MLFlow Tracking Server

To use a locally hosted MLflow server to view the experiments stored in mlruns, do the following steps.

### Start a Local MLflow Server
If you don't have MLflow installed, please run the command below to install it:

``` python
pip install mlflow
```

The installation of MLflow includes the MLflow CLI tool, so you can start a local MLflow server with UI by running the command below in your terminal:

``` python
mlflow ui
```

### View Experiment on Your MLflow Server
Now let's view your experiment on the local server. Open the URL in your browser, which is http://localhost:5000 in our case.
