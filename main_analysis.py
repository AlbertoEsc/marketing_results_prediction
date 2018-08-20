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
from sklearn.metrics import log_loss as cross_entropy

# Functions and classes used for hyperparameter search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
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
from data_loading import compute_indices_train_val_test
from data_loading import split_data
from null_handling import remove_nulls
from correlation_analysis import analyze_using_PCA
from correlation_analysis import analyze_correlation_matrix

# Configuration. These variables specify
# the execution of the whole analysis
filename = 'data/testdata.csv'
explanations_filename = 'data/variable_explanations.csv'
NUMPY_SEED = 1234
enable_correlation_analysis = False
enable_pca_analysis = False

enable_random_forest_classifier = True
enable_gradient_boosting_classifier = False # or True

enable_ridge_regression = False #or True
enable_random_forest_regressor = False or True
enable_gradient_boosting_regressor = False #or True

# Load data and field explanations (files have been renamed)
data_df, id_df, target_sold_df, target_sales_df = read_data(filename)
print(data_df.describe())
print(target_sold_df.describe())
print(target_sales_df.describe())
data_df = remove_nulls(data_df, mode='mean', add_null_columns=True)

# print("data_df.columns[50]:", data_df.columns[50])
# print("data_df.columns[61]:", data_df.columns[61])

explanations = read_explanations(explanations_filename)
print("Feature descriptions:\n", explanations)

if enable_correlation_analysis:
    analyze_correlation_matrix(target_sold_df, target_sales_df, data_df)

# Convert data to numpy
lead_id = id_df.values
target_sold = target_sold_df.values
target_sales_all = np.nan_to_num(target_sales_df.values)
data = data_df.values.astype(float)
print('data.shape:', data.shape)

# Apply correlation analysis
if enable_pca_analysis:
    analyze_using_PCA(target_sold, target_sales_all, data)
# quit()

# Extract data for sales predictions
data_sales_sel = data[target_sold == 1.0, :]
target_sales_sel = target_sales_all[target_sold == 1.0]
print("len(data_sales_sel):", len(data_sales_sel),
      "len(target_sales_sel):", len(target_sales_sel))

# Label enlargement

# Split of samples (train, validation, test)
# The features for the first model are called X_train, X_val, X_test,
# and labels (target_sold) are called y_train, y_val, y_test
np.random.seed(NUMPY_SEED)
indices_train, indices_val, indices_test = \
    compute_indices_train_val_test(len(data), frac_val=0.2, frac_test=0.2)
X_train, X_val, X_test = \
    split_data(data, indices_train, indices_val, indices_test)
y_train, y_val, y_test = \
    split_data(target_sold, indices_train, indices_val, indices_test)

# Data for second model
X_m2_train = X_train[y_train == 1]
X_m2_val = X_val[y_val == 1]
X_m2_test = X_test[y_test == 1]
y_m2_train = target_sales_all[indices_train][y_train == 1]
y_m2_val = target_sales_all[indices_val][y_val == 1]
y_m2_test = target_sales_all[indices_test][y_test == 1]

# Data for third model
X_m3_train = X_train
X_m3_val = X_val
X_m3_test = X_test
y_m3_train = target_sales_all[indices_train]
y_m3_val = target_sales_all[indices_val]
y_m3_test = target_sales_all[indices_test]

# TODO: consider aligning the data for both models
# RANDOM_STATE_TEST = 12
# RANDOM_STATE_VAL = 56
# X_m2_train, X_m2_test, y_m2_train, y_m2_test = \
#         train_test_split(data_sales_sel, target_sales_sel, test_size=0.20,
#                          random_state=RANDOM_STATE_TEST)
# X_m2_train, X_m2_val, y_m2_train, y_m2_val = \
#         train_test_split(X_m2_train, y_m2_train, test_size=0.25,
#                          random_state=RANDOM_STATE_VAL)
print("len(X_m2_train):", len(X_m2_train),
      "len(X_m2_val):", len(X_m2_val),
      "len(X_m2_test):", len(X_m2_test))
print("len(y_m2_train):", len(y_m2_train),
      "len(y_m2_val):", len(y_m2_val),
      "len(y_m2_test):", len(y_m2_test))


# Normalize the data
# first model
X_train_mean = X_train.mean(axis=0)
print("X_train:", X_train)
print("X_train[0]:", X_train[0])
print("X_train.shape:", X_train.shape)
X_train_std = X_train.std(axis=0)
print("X_train_std:", X_train_std)
X_train_std += 0.01
X_train =  (X_train - X_train_mean) / X_train_std
X_val =  (X_val - X_train_mean) / X_train_std
X_test =  (X_test - X_train_mean) / X_train_std
# second model uses the same normalization
#X_m2_train_mean = X_m2_train.mean(axis=0)
#X_m2_train_std = X_m2_train.std(axis=0)
X_m2_train =  (X_m2_train - X_train_mean) / X_train_std
X_m2_val =  (X_m2_val - X_train_mean) / X_train_std
X_m2_test =  (X_m2_test - X_train_mean) / X_train_std





# Removal of outliers from training data
#y = target_sales_all[target_sold == 1.0].copy()
#y = np.sort(y)
#print(y, y.shape)
#print("top 10 sales:", y[-10:-5])
#print("top 10 sales:", y[-5:])
#quit()

# Feature normalization (zero-mean unit-variance or range [0, 1])

# ML methods
# First model (M1). Estimation of whether a lead will result in 'sold'
test_acc = {}
classification_methods = []
# M1. ChanceLevel estimations
test_acc[('ChanceLevel_MSE', 'train')] = mean_squared_error(y_train, y_train.mean() * np.ones_like(y_train))
test_acc[('ChanceLevel_MSE', 'val')] = mean_squared_error(y_val, y_val.mean() * np.ones_like(y_val))
test_acc[('ChanceLevel_CR', 'train')] = accuracy_score(y_train, stats.mode(y_train)[0] * np.ones_like(y_train))
test_acc[('ChanceLevel_CR', 'val')] = accuracy_score(y_val, stats.mode(y_val)[0] * np.ones_like(y_val))

# M1. LinearRegression
lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
lr.fit(X_train, y_train)
pred_lr_train = lr.predict(X_train)
pred_lr_train = pred_lr_train.clip(0, 1)
prob_lr_train = np.stack((1-pred_lr_train, pred_lr_train), axis=1)
pred_lr_val = lr.predict(X_val).clip(0, 1)
pred_lr_val = pred_lr_val.clip(0, 1)
prob_lr_val = np.stack((1-pred_lr_val, pred_lr_val), axis=1)
test_acc[('LinearRegression_MSE', 'train')] = mean_squared_error(y_train, pred_lr_train)
test_acc[('LinearRegression_MSE', 'val')] = mean_squared_error(y_val, pred_lr_val)
test_acc[('LinearRegression_CR', 'train')] = accuracy_score(y_train, pred_lr_train >= 0.5)
test_acc[('LinearRegression_CR', 'val')] = accuracy_score(y_val, pred_lr_val >= 0.5)
test_acc[('LinearRegression_CE', 'train')] = cross_entropy(y_train, prob_lr_train)
test_acc[('LinearRegression_CE', 'val')] = cross_entropy(y_val, prob_lr_val)
print(lr.coef_)
classification_methods.append('lr')

# M1. LogisticRegression
logr = LogisticRegression(C=1.0)
logr.fit(X_train, y_train)
pred_logr_train = logr.predict(X_train)
pred_logr_val = logr.predict(X_val)
prob_logr_train = logr.predict_proba(X_train)
prob_logr_val = logr.predict_proba(X_val)
test_acc[('LogisticRegression_MSE', 'train')] = mean_squared_error(y_train, pred_logr_train)
test_acc[('LogisticRegression_MSE', 'val')] = mean_squared_error(y_val, pred_logr_val)
test_acc[('LogisticRegression_CR', 'train')] = accuracy_score(y_train, pred_logr_train)
test_acc[('LogisticRegression_CR', 'val')] = accuracy_score(y_val, pred_logr_val)
test_acc[('LogisticRegression_CE', 'train')] = cross_entropy(y_train, prob_logr_train)
test_acc[('LogisticRegression_CE', 'val')] = cross_entropy(y_val, prob_logr_val)
classification_methods.append('logr')

# M1. RandomForestClassifier
if enable_random_forest_classifier:
    print('Using randomized search to tune a random forest classifier...')
    param_dist_rf = {"n_estimators": sp_randint(15, 25)}
    rf = RandomForestClassifier()
    rf_rs = RandomizedSearchCV(rf, n_iter=3, cv=5,
                               param_distributions=param_dist_rf)
    rf_rs.fit(X_train, y_train)
    pred_rf_train = rf_rs.predict(X_train)
    pred_rf_val = rf_rs.predict(X_val)
    prob_rf_train = rf_rs.predict_proba(X_train)
    prob_rf_val = rf_rs.predict_proba(X_val)
    test_acc[('RandomForestClassifier_CR', 'train')] = accuracy_score(y_train, pred_rf_train)
    test_acc[('RandomForestClassifier_CR', 'val')] = accuracy_score(y_val, pred_rf_val)
    test_acc[('RandomForestClassifier_CE', 'train')] = cross_entropy(y_train, prob_rf_train)
    test_acc[('RandomForestClassifier_CE', 'val')] = cross_entropy(y_val, prob_rf_val)
    print("RandomForest best_score:", rf_rs.best_score_,
          "best_params:", rf_rs.best_params_)
    classification_methods.append('rfc')

# M1. GradientBoostingClassifier
if enable_gradient_boosting_classifier:
    print('Using randomized search to tune a gradient boosting classifier...')
    param_dist_gbc = {"n_estimators": sp_randint(105, 145),
                      "max_depth": sp_randint(2,5),
                      "learning_rate": sp_uniform(0.55, 0.75)}
    gbc = GradientBoostingClassifier()
    gbc_rs = RandomizedSearchCV(gbc, n_iter=1, cv=5,
                                param_distributions=param_dist_gbc)
    gbc_rs.fit(X_train, y_train)
    pred_gbc_train = gbc_rs.predict(X_train)
    pred_gbc_val = gbc_rs.predict(X_val)
    prob_gbc_train = gbc_rs.predict_proba(X_train)
    prob_gbc_val = gbc_rs.predict_proba(X_val)
    test_acc[('GradientBoostingClassifier_CR', 'train')] = \
        accuracy_score(y_train, pred_gbc_train)
    test_acc[('GradientBoostingClassifier_CR', 'val')] = \
        accuracy_score(y_val, pred_gbc_val)
    test_acc[('GradientBoostingClassifier_CE', 'train')] = \
        cross_entropy(y_train, pred_gbc_train)
    test_acc[('GradientBoostingClassifier_CE', 'val')] = \
        cross_entropy(y_val, pred_gbc_val)

    print("GradientBoostingClassifier best_score:", gbc_rs.best_score_,
          "best_params:", gbc_rs.best_params_)
    classification_methods.append('gbc')

print(test_acc)

res_sold_df = pd.DataFrame()
for (alg, test_set) in test_acc.keys():
    res_sold_df.loc[alg, test_set] = test_acc[(alg, test_set)]
print(res_sold_df)

# Second model (M2). Estimation of sales for successful leads
test_m2_acc = {}
regression_methods = []
# Chance levels
test_m2_acc[('ChanceLevel_MSE', 'train')] = \
    mean_squared_error(y_m2_train, y_m2_train.mean() * np.ones_like(y_m2_train))
test_m2_acc[('ChanceLevel_MSE', 'val')] = \
    mean_squared_error(y_m2_val, y_m2_val.mean() * np.ones_like(y_m2_val))

# M2. LinearRegression
lr_m2 = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
lr_m2.fit(X_m2_train, y_m2_train)
pred_lr_train = lr_m2.predict(X_m2_train)
pred_lr_val = lr_m2.predict(X_m2_val)
test_m2_acc[('LinearRegression_MSE', 'train')] = mean_squared_error(y_m2_train, pred_lr_train)
test_m2_acc[('LinearRegression_MSE', 'val')] = mean_squared_error(y_m2_val, pred_lr_val)
print(lr_m2.coef_)
regression_methods.append('lr')

# M2. Ridge regression
if enable_ridge_regression:
    print('Using randomized search to tune ridge regression...')
    param_dist_rid = {"alpha": sp_uniform(0,3)}
    rid = Ridge(alpha=0.125, normalize=True)
    rid_rs = RandomizedSearchCV(rid, n_iter=3, cv=5,
                                param_distributions=param_dist_rid)
    rid_rs.fit(X_m2_train, y_m2_train)
    pred_rid_train = rid_rs.predict(X_m2_train)
    pred_rid_val = rid_rs.predict(X_m2_val)
    test_m2_acc[('Ridge_MSE', 'train')] = mean_squared_error(y_m2_train, pred_rid_train)
    test_m2_acc[('Ridge_MSE', 'val')] = mean_squared_error(y_m2_val, pred_rid_val)
    print("Ridge best_score:", rid_rs.best_score_,
          "best_params:", rid_rs.best_params_)
    regression_methods.append('rid')

# M2. RandomForestRegressor
if enable_random_forest_regressor:
    print('Using randomized search to tune a random forest regressor...')
    param_dist_rfr = {"max_depth": [1, 2, 3, 4, 5]}
    rfr = RandomForestRegressor(max_depth=3, random_state=0)
    rfr_rs = RandomizedSearchCV(rfr, n_iter=5, cv=5,
                                param_distributions=param_dist_rfr)
    rfr_rs.fit(X_m2_train, y_m2_train)
    pred_rfr_train = rfr_rs.predict(X_m2_train)
    pred_rfr_val = rfr_rs.predict(X_m2_val)
    test_m2_acc[('RandomForestRegressor_MSE', 'train')] = mean_squared_error(y_m2_train, pred_rfr_train)
    test_m2_acc[('RandomForestRegressor_MSE', 'val')] = mean_squared_error(y_m2_val, pred_rfr_val)
    print("RandomForestRegressor best_score:", rfr_rs.best_score_,
          "best_params:", rfr_rs.best_params_)
    regression_methods.append('rfr')

# M2. GradientBoostingRegressor
if enable_gradient_boosting_regressor:
    print('Using randomized search to tune a gradient boosting regressor...')
    param_dist_gbr = {"n_estimators": sp_randint(50, 110),
                      "max_depth": sp_randint(1, 3),
                      "learning_rate": sp_uniform(0.05, 0.25)}
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls')
    gbr_rs = RandomizedSearchCV(gbr, n_iter=1, cv=5,
                                param_distributions=param_dist_gbr)
    gbr_rs.fit(X_m2_train, y_m2_train)
    pred_gbr_train = gbr_rs.predict(X_m2_train)
    pred_gbr_val = gbr_rs.predict(X_m2_val)
    test_m2_acc[('GradientBoostingRegressor_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_gbr_train)
    test_m2_acc[('GradientBoostingRegressor_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_gbr_val)
    print("GradientBoostingRegressor best_score:", gbr_rs.best_score_,
          "best_params:", gbr_rs.best_params_)
    regression_methods.append('gbr')

res_sales_df = pd.DataFrame()
for (alg, test_set) in test_m2_acc.keys():
    res_sales_df.loc[alg, test_set] = test_m2_acc[(alg, test_set)]
print(res_sales_df)

# Third model (M3). Direct computation of expected return
# M3. RandomForestRegressor
enable_random_forest_regressor_m3 = True
test_m3_acc = {}
if enable_random_forest_regressor_m3:
    print('Using randomized search to tune a random forest regressor...')
    param_dist_rfr_m3 = {"max_depth": [1, 2, 3, 4, 5]}
    rfr_m3 = RandomForestRegressor(max_depth=3, random_state=0)
    rfr_rs_m3 = RandomizedSearchCV(rfr_m3, n_iter=5, cv=5,
                                param_distributions=param_dist_rfr_m3)
    rfr_rs_m3.fit(X_m3_train, y_m3_train)
    pred_rfr_train_m3 = rfr_rs_m3.predict(X_m3_train)
    pred_rfr_val_m3 = rfr_rs_m3.predict(X_m3_val)
    test_m3_acc[('RandomForestRegressor_MSE', 'train')] = mean_squared_error(y_m3_train, pred_rfr_train_m3)
    test_m3_acc[('RandomForestRegressor_MSE', 'val')] = mean_squared_error(y_m3_val, pred_rfr_val_m3)
    print("RandomForestRegressor best_score:", rfr_rs_m3.best_score_,
          "best_params:", rfr_rs_m3.best_params_)
res_sales_m3_df = pd.DataFrame()
for (alg, test_set) in test_m3_acc.keys():
    res_sales_m3_df.loc[alg, test_set] = test_m3_acc[(alg, test_set)]
print(res_sales_m3_df)

# Expected sales (prob of sale * estimated sale value)
# pred_m2_lr_train_all = lr_m2.predict(X_train)
# pred_m2_lr_val_all = lr_m2.predict(X_val)
# pred_m2_lr_test_all = lr_m2.predict(X_test)

total_revenue_train = target_sales_all[indices_train].sum()
total_revenue_val = target_sales_all[indices_val].sum()
print("total_revenue_train", total_revenue_train)
print("total_revenue_val", total_revenue_val)

classification_method = 'rfc'
regression_method = 'rfr'
leads_kept_train = np.rint(0.40 * len(X_train)).astype(int)
leads_kept_val = np.rint(0.40 * len(X_val)).astype(int)

for classification_method in classification_methods:
    if classification_method == 'rfc':
        prob_train = prob_rf_train[:,1]
        prob_val = prob_rf_val[:,1]
    elif classification_method == 'logr':
        prob_train = prob_logr_train[:,1]
        prob_val = prob_logr_val[:,1]
    elif classification_method == 'lr':
        prob_train = prob_lr_train[:,1]
        prob_val = prob_lr_val[:,1]
    elif classification_method == 'gbc':
        prob_train = prob_gbc_train[:, 1]
        prob_val = prob_gbc_val[:, 1]
    else:
        raise ValueError('Unknown classification_method:', classification_method)

    for regression_method in regression_methods:
        if regression_method == 'lr':
            pred_train = lr_m2.predict(X_train)
            pred_val = lr_m2.predict(X_val)
        elif regression_method == 'rid':
            pred_train = rid_rs.predict(X_train)
            pred_val = rid_rs.predict(X_val)
        elif regression_method == 'rfr':
            pred_train = rfr_rs.predict(X_train)
            pred_val = rfr_rs.predict(X_val)
        elif regression_method == 'gbr':
            pred_train = gbr_rs.predict(X_train)
            pred_val = bgr_rs.predict(X_val)
        else:
            raise ValueError('Unknown regression_method:', regression_method)

        expected_sales_train = prob_train * pred_train
        expected_sales_val = prob_val * pred_val

        #print("expected_sales_train:\n", expected_sales_train)
        #print("expected_sales_train_m3:\n", expected_sales_train_m3)
        #print("true sales_train:\n", target_sales_all[indices_train])
        #print("expected_sales_val:\n", expected_sales_val)
        #print("expected_sales_val_m3:\n", expected_sales_val_m3)
        #print("true sales_val:\n", target_sales_all[indices_val])

        promising_sales_indices_train = np.argsort(expected_sales_train)
        promising_sales_indices_val = np.argsort(expected_sales_val)

        top_promising_sales_indices_train = promising_sales_indices_train[-leads_kept_train:]
        top_promising_sales_indices_val = promising_sales_indices_val[-leads_kept_val:]

        #print("top expected_sales_train:\n", expected_sales_train[top_promising_sales_indices_train])
        #print("correct expected_sales_train:\n", target_sales_all[indices_train][top_promising_sales_indices_train])
        #print("top expected_sales_val:\n", expected_sales_val[top_promising_sales_indices_val])
        #print("correct expected_sales_val:\n", target_sales_all[indices_val][top_promising_sales_indices_val])

        print("classification:", classification_method, "regression:", regression_method)
        e_total_revenue_train_sel = expected_sales_train[top_promising_sales_indices_train].sum()
        e_total_revenue_val_sel = expected_sales_val[top_promising_sales_indices_val].sum()
        print("e_total_revenue_train_sel", e_total_revenue_train_sel)
        print("e_total_revenue_val_sel", e_total_revenue_val_sel)

        total_revenue_train_sel = target_sales_all[indices_train][top_promising_sales_indices_train].sum()
        total_revenue_val_sel = target_sales_all[indices_val][top_promising_sales_indices_val].sum()
        print("total_revenue_train_sel", total_revenue_train_sel)
        print("total_revenue_val_sel", total_revenue_val_sel)


expected_sales_train_m3 = pred_rfr_train_m3
expected_sales_val_m3 = pred_rfr_val_m3
promising_sales_indices_train_m3 = np.argsort(expected_sales_train_m3)
promising_sales_indices_val_m3 = np.argsort(expected_sales_val_m3)
top_promising_sales_indices_train_m3 = promising_sales_indices_train_m3[-leads_kept_train:]
top_promising_sales_indices_val_m3 = promising_sales_indices_val_m3[-leads_kept_val:]

e_total_revenue_train_sel_m3 = expected_sales_train_m3[top_promising_sales_indices_train_m3].sum()
e_total_revenue_val_sel_m3 = expected_sales_val_m3[top_promising_sales_indices_val_m3].sum()
print("e_total_revenue_train_sel_m3", e_total_revenue_train_sel_m3)
print("e_total_revenue_val_sel_m3", e_total_revenue_val_sel_m3)

total_revenue_train_sel_m3 = target_sales_all[indices_train][top_promising_sales_indices_train_m3].sum()
total_revenue_val_sel_m3 = target_sales_all[indices_val][top_promising_sales_indices_val_m3].sum()
print("total_revenue_train_sel_m3", total_revenue_train_sel_m3)
print("total_revenue_val_sel_m3", total_revenue_val_sel_m3)


# TODO: Create module for data_loading
# TODO: Add baseline model for target_sales
# TODO: Add more algorithms
# TODO: Hyperparameter search


# TODO: use best hyperparameters and repeat training on data (train+val)