########################################################################
# Machine learning task 2                                              #
#                                                                      #
# Prediction of lead success and transaction price                     #
#                                                                      #
# Author: Alberto N. Escalante B.                                      #
# Date: 13.08.2018                                                     #
# E-mail: alberto.escalante@ini.rub.de                                 #
#                                                                      #
########################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# Regression/ensemble algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Classification/ensemble algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

from data_loading import read_data
from data_loading import read_explanations
from null_handling import remove_nulls

# Load data and field explanations (files have been renamed)
filename = 'data/testdata.csv'
data_df, id_df, target_sold_df, target_sales_df = read_data(filename)
data_df = remove_nulls(data_df, mode='mean', add_null_columns=True)

explanations_filename = 'data/variable_explanations.csv'
explanations = read_explanations(explanations_filename)
print("Feature descriptions:\n", explanations)

# Convert data to numpy
lead_id = id_df.values
target_sold = target_sold_df.values
target_sales_all = target_sales_df.values
data = data_df.values
print('data.shape:', data.shape)

# Removal of outliers

# Basic experiment to see correlations
pca = PCA(n_components=30) # TODO: center and normalize data
pca.fit(data)
y = pca.transform(data)
print(data[0:1,:])
print(pca.inverse_transform(y[0:1,:]))
print('pca.explained_variance_:', pca.explained_variance_)
print('pca.components_:', pca.components_)
print('pca.components_[0]:', pca.components_[0])

# Extract data for sales predictions
data_sales_sel = data[target_sold == 1.0, :]
target_sales_sel = target_sales_all[target_sold == 1.0]
print("len(data_sales_sel):", len(data_sales_sel),
      "len(target_sales_sel):", len(target_sales_sel))

# Label enlargement

# Split of samples (train, validation, test)
# The features for the first model are called X_train, X_val, X_test,
# and labels (target_sold) are called y_train, y_val, y_test
RANDOM_STATE_TEST = 1234
RANDOM_STATE_VAL = 5678
X_train, X_test, y_train, y_test = \
        train_test_split(data, target_sold, test_size=0.20,
                         random_state=RANDOM_STATE_TEST)
X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.25,
                         random_state=RANDOM_STATE_VAL)
print("len(X_train):", len(X_train),
      "len(X_val):", len(X_val),
      "len(X_test):", len(X_test))
print("len(y_train):", len(y_train),
      "len(y_val):", len(y_val),
      "len(y_test):", len(y_test))

# TODO: consider aligning the data for both models
X_m2_train, X_m2_test, y_m2_train, y_m2_test = \
        train_test_split(data_sales_sel, target_sales_sel, test_size=0.20,
                         random_state=RANDOM_STATE_TEST)
X_m2_train, X_m2_val, y_m2_train, y_m2_val = \
        train_test_split(X_m2_train, y_m2_train, test_size=0.25,
                         random_state=RANDOM_STATE_VAL)
print("len(X_m2_train):", len(X_m2_train),
      "len(X_m2_val):", len(X_m2_val),
      "len(X_m2_test):", len(X_m2_test))
print("len(y_m2_train):", len(y_m2_train),
      "len(y_m2_val):", len(y_m2_val),
      "len(y_m2_test):", len(y_m2_test))

# ML methods
test_acc = {}
# M1. ChanceLevel
test_acc[('ChanceLevel_MSE', 'train')] = mean_squared_error(y_train, y_train.mean() * np.ones_like(y_train))
test_acc[('ChanceLevel_MSE', 'val')] = mean_squared_error(y_val, y_val.mean() * np.ones_like(y_val))
test_acc[('ChanceLevel_CR', 'train')] = accuracy_score(y_train, stats.mode(y_train)[0] * np.ones_like(y_train))
test_acc[('ChanceLevel_CR', 'val')] = accuracy_score(y_val, stats.mode(y_val)[0] * np.ones_like(y_val))

# M1. LinearRegression
lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
lr.fit(X_train, y_train)
pred_lr_train = lr.predict(X_train)
pred_lr_val = lr.predict(X_val)
test_acc[('LinearRegression_MSE', 'train')] = mean_squared_error(y_train, pred_lr_train)
test_acc[('LinearRegression_MSE', 'val')] = mean_squared_error(y_val, pred_lr_val)
test_acc[('LinearRegression_CR', 'train')] = accuracy_score(y_train, pred_lr_train >= 0.5)
test_acc[('LinearRegression_CR', 'val')] = accuracy_score(y_val, pred_lr_val >= 0.5)
print(lr.coef_)

logr = LogisticRegression(C=1.0)
logr.fit(X_train, y_train)
pred_logr_train = logr.predict(X_train)
pred_logr_val = logr.predict(X_val)
test_acc[('LogisticRegression_MSE', 'train')] = mean_squared_error(y_train, pred_logr_train)
test_acc[('LogisticRegression_MSE', 'val')] = mean_squared_error(y_val, pred_logr_val)
test_acc[('LogisticRegression_CR', 'train')] = accuracy_score(y_train, pred_logr_train)
test_acc[('LogisticRegression_CR', 'val')] = accuracy_score(y_val, pred_logr_val)
print(test_acc)

res_sold_df = pd.DataFrame()
for (alg, test_set) in test_acc.keys():
    res_sold_df.loc[alg, test_set] = test_acc[(alg, test_set)]
print(res_sold_df)


#Second model
test_m2_acc = {}
test_m2_acc[('ChanceLevel_MSE', 'train')] = \
    mean_squared_error(y_m2_train, y_m2_train.mean() * np.ones_like(y_m2_train))
test_m2_acc[('ChanceLevel_MSE', 'val')] = \
    mean_squared_error(y_m2_val, y_m2_val.mean() * np.ones_like(y_m2_val))

lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
lr.fit(X_m2_train, y_m2_train)
pred_lr_train = lr.predict(X_m2_train)
pred_lr_val = lr.predict(X_m2_val)
test_m2_acc[('LinearRegression_MSE', 'train')] = mean_squared_error(y_m2_train, pred_lr_train)
test_m2_acc[('LinearRegression_MSE', 'val')] = mean_squared_error(y_m2_val, pred_lr_val)
print(lr.coef_)

res_sales_df = pd.DataFrame()
for (alg, test_set) in test_m2_acc.keys():
    res_sales_df.loc[alg, test_set] = test_m2_acc[(alg, test_set)]
print(res_sales_df)

#TODO: Create module for data_loading
#TODO: Add baseline model for target_sales
#TODO: Add more algorithms
#TODO: Hyperparameter search
