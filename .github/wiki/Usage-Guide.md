The current package aims at predicting a `target_variable` from a set of 2 input variables from a very simple Machine Learning algorithm.


# 1. Working with the `config.yaml` file

For this the `config.yaml` should be properly edited:
* files
    * `train`: the name of the filen used for training the algorithm (should be a CSV file for the basic training script can be CSV or JSON for the dynamic training script)
    * `test`: the name of the file used for predictions
* floders:
    * `data`: the relative path to the data folder containng the train/test files
    * `models`: the relative path of the folder used to store the trained model
* mlflow:
    * `experiment_name`: the name for the MLFlow experiments
    * `tracking_server`: a boolean to activate MLFlow tracking server (if True, make sure tracking server is running)
    * `uri`: the URI of the tracking server

* `n_data_points`: Number of data points to use for dynamic training (must be >= 4)

The training (fixed or dynamic from the train file ) and predicitions (from the test file) can be run with simple `make` commands:
``` bash
make train
make train-dynamic
make predict
```

# 2. Working with the CLI
Additionnally the script can be run and the arguments can be called from the
command line interface.

``` bash
python src/watergenic_coding_test/train.py -i data/202502_data_train.csv
python src/watergenic_coding_test/train_dynamic.py -i data/202502_data_train.json -n 8

python src/watergenic_coding_test/predict.py -i data/202302_data_test.csv
```
