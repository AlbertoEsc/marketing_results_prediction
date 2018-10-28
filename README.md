# Marketing Results Prediction

*Marketing Results Prediction* is an exercise/project that uses supervised learning methods to understand and predict the success of a marketing strategy on individual customers.
The resulting models allow the sales team to reduce their marketing efforts by focusing on the most promising customers, who are identified based on their expected return.

## Data
The data is contained in a single CSV file that describes previous marketing results on different customers. Each row represents a customer and the different interactions that have already taken place. There are two important fields (i.e., variables/predictors) within each row: *contract closed* (a binary variable) and *contract value* (a non-negative float value). These two fields are the labels that are predicted. The semantics of the remaining approx. 50 fields are not disclosed, but most of  them are indeed useful to estimate the labels.

## Solution
In order to estimate the expected contract value as accurately as possible, we first compute the probability that a contract is closed and then estimate the contract value. More concretely, three models are built for the proposed solution.

* Model 1 (M1) is a classification model that predicts the probability that 'contract closed' is 1.
* Model 2 (M2) is a regression model that predicts the variable 'contract value'. The data used to train this model is restricted to samples for which **'contract closed' is 1.**
* Model 3 (M3) is a regression model that allows the prediction of the variable 'contract value' for any sample. If for a particular sample the variable 'contract closed' is 0, the corresponding variable 'contract value' is assumed to be zero.

Model 1 and 2 are combined in a single model that predicts the *expected return* for an arbitrary customer (the expected return is simply the product of the predictions of model 1 and model 2). This is better than simply focusing on the samples with highest probability of a contract and also better than focusing on the samples with largest predicted contract value.

## Algorithms
For each one of the models, different algorithms with various complexities were tested, including basic ones (such as linear regression and logistic regression) and more powerful ones (e.g., gradient boosting, random forests, and a 4-layer neural network). The implementations used are those from sklearn and tf.estimators. In all cases *randomized search with cross validation* was used to select good hyperparameters.
The algorithms used for model 1 are:

* Logistic Regression
* K-Nearest Neighbors Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Classifier (SVC) with an RBF kernel
* A 4-layer neural network (DNNClassifier). For this method hyperparameter search is also done via random search, but a single validation set is used instead of cross validation)

The algorithms used for model 2 are:
* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regression (SVR) with an RBF kernel
* A 4-layer neural network (DNNRegressor). Hyperparameter search is done as in the DNNClassifier


The algorithms used for model 3 are:

* Random Forest Regressor
* Support Vector Regression (SVR) with an RBF kernel

Moreover, in order to compute *expected return values*, all possible combinations of algorithms for models M1 and M2 were tested. According to the validation data, currently the best combination of methods is the Random Forest Classifier (M1) with the Random Forest Regressor (M2). Furthermore, this combination of methods is more accurate than the direct estimation of expected return values by model M3.

## Dependencies
Marketing Results Prediction requires the following libraries:

* numpy
* pandas
* tensorflow (the code has been tested on TF versions 1.8 and 1.11)
* scipy
* sklearn
* matplotlib
* csv


## Usage and further documentation
The code can be simply run as:
  > python3 -u main_analysis.py

Several Boolean flags control which algorithms are executed and which plots are generated. Such flags can be set directly in the source code (see 'configuration variables' section in main_analysis.py).


## Author
Marketing Results Prediction has been developed by Alberto N. Escalante B. (alberto.escalante@ini.rub.de) as a data science exercise/project. It might be useful as a code example to learn methods of Data Science that use pandas/numpy/sklearn/TensorFlow.

Bugs/Suggestions/Comments/Questions: please send them to alberto.escalante@ini.rub.de or via github.
I will be glad to help you.
