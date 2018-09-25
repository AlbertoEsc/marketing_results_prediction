# Marketing Results Prediction

Marketing Results Prediction is a Data Science exercise where a few problems are solved that allow the sales team to focus their marketing efforts in a few customers.

## Data
The data is contained in a CSV file, where each sample (row) contains two important labels: contract closed (a binary variable) and sales value (a non-negative float value). The semantics of the remaining about 50 variables are not known, but (most of) these variables are useful to estimate the two labels.
** The data is not yet available publicly **

## Solution 
The proposed solution builds three different models.
* Model 1 (M1) is a classification model that predicts the probability that 'contract closed' is 1.
* Model 2 (M2) is a regression model that predicts the variable 'sales value' **only when a 'contract closed' is 1.**
* Model 3 (M3) is a regression model that predicts the variable 'sales value'. If 'contract closed' is 0, the variable 'sales value' is assumed to be zero.

## Algorithms
Different algorithms were tested for each model, including basic algorithms and more challenging ones. The implementations used are those from sklearn and tf.estimators. In all cases hyperparameters where found using randomized search with cross validation.
The algorithms used for model 1 are: 
* Logistic Regression
* K-Nearest Neighbors Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Classifier (SVC)

The algorithms used for model 2 are: 
* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regression (SVR)
* A 3-layer neural network (DNNClassifier, without cross validation)

The algorithms used for model 3 are:
* Random Forest Regressor
* Support Vector Regression (SVR)


## Dependencies
Marketing Results Prediction requires the following libraries:
* numpy
* pandas
* tensorflow (the code runs correctly at least on version 1.8)
* scipy
* sklearn
* matplotlib
* csv


## Usage and further documentation
The code can be simply run as:
  > python3 -u main_analysis.py

Several options control which algorithms are executed. Such options need to be changed directly in the source code (Congiguration Variables section).


## Author
Marketing Results Prediction has been developed by Alberto N. Escalante B. (alberto.escalante@ini.rub.de) as a data science exercise/project. It can be useful to learn some methods of Data Science using the pandas/numpy/sklearn/TensorFlow libraries.

Bugs/Suggestions/Comments/Questions: please send them to alberto.escalante@ini.rub.de or via github.
I will be glad to help you.
