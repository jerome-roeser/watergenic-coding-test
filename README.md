# 🧪 Watergenics Candidate Coding Test – Machine Learning Mini Project (AI-Enhanced)


## 🎯 Objective
This project evaluates your **Python and ML skills**, your ability to **plan and document tasks**, and your **understanding and responsible use of AI tools**.
You will implement a small ML project using `scikit-learn`, `MLflow`, and `unit testing`— while demonstrating strong skills in problem solving, progress documentation, and optionally using AI tools responsibly.


## 💬 My approach in 2 words
Before coding the project / package, I performed exploratory data analysis (EDA) in order
to get a hand on the data and decide which ML model and features to use to predict the target.

This EDA step also allowed me to choose the metrics to evaluate the performance of the model.


### Conclusions and assumptions made

* the dataset is very small and clean (no NaN or duplicates)
* We assume the time is not a feature so no feature engineering is made with this data
* it appears wuite clearly that the variance of the `target_variable` is explained by the variance of the `input_variable1`
* the **coefficient of determination** (R2) is between the 2 variables is almost 1
* thus a simple **Linear Regression** on `input_variable1` will be used as ML algorithm
* seeing the range of values of the target and the high R2, I will use 2 metrics to evaluate the model:
    * **R2**
    * **mean absolute percentage error**:  which will give us a good indication of the performance of the model in that case

### ❗ Issue
It seems that the model  doesn't perform that well on the testing set.
* Three main reasons can be thought of to explain this behaviour:
    * the model is **not complex enough**: but adding the 2nd feature and testing other
models didn't really change the performance
    * the **low number of training points** limits the ability to capture the variance of the target variable
    * there is a data drift and the training and test datasets might have different distributions

✅ To prepare a better model and make better predicitions, a few approaches can be exploited
* **Collect more data** and more features or do feature engineering
* Do an **Adversarial Validation** by training a classifier an the training and testing set
