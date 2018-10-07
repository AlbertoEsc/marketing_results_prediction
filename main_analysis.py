########################################################################
# Data Science Exercise.                                               #
#                                                                      #
# Prediction of contract probability and contract value                #
#                                                                      #
# Author: Alberto N. Escalante B.                                      #
# Date: 13.08.2018                                                     #
# E-mail: alberto.escalante@ini.rub.de                                 #
#                                                                      #
########################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
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
from sklearn.model_selection import ParameterSampler
# Regression/ensemble algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
# Classification/ensemble algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# Plots
import matplotlib.pyplot as plt

from data_loading import read_data
from data_loading import read_explanations
from data_loading import compute_indices_train_val_test
from data_loading import split_data
from null_handling import remove_nulls
from correlation_analysis import analyze_using_PCA
from correlation_analysis import analyze_correlation_matrix
from tensorflow_data_processing import train_input_fn
from tensorflow_data_processing import eval_input_fn
from tensorflow_data_processing import extract_pred_and_prob_from_estimator_predictions

############################################################################
# Configuration. These variables specify what parts of
# the whole analysis are executed
filename = 'data/contract_data.csv'
explanations_filename = 'data/variable_explanations.csv'
NUMPY_SEED = 12345
enable_feature_selection = False # or True
enable_histograms = False # or True
enable_correlation_analysis = False # or True
enable_pca_analysis = False  # or True

enable_logistic_regression = False 
enable_linear_regression_for_sold = False 
enable_random_forest_classifier = True
enable_gradient_boosting_classifier = False 
enable_support_vector_classifier = False 
enable_dnn_classifier = True

enable_linear_regression_for_sales = False 
enable_ridge_regression = False 
enable_random_forest_regressor = True
enable_gradient_boosting_regressor = False 
enable_support_vector_regressor = False 

enable_random_forest_regressor_m3 = False or True
enable_support_vector_regressor_m3 = False

verbose = False
evaluate_test_data = False  # or True
enable_efficiency_plot = False

############################################################################
# Data loading and explorative data analysis
# Load data and field explanations (files have been renamed)
data_df, id_df, target_sold_df, target_sales_df = read_data(filename)
print(data_df.describe())
print(target_sold_df.describe())
print(target_sales_df.describe())

if enable_feature_selection:
    all_variables_by_importance = []
    num_vars = len(all_variables_by_importance)
    num_dropped_vars = 4
    dropped_variables = all_variables_by_importance[num_vars -
                                                    num_dropped_vars:num_vars]
    if len(dropped_variables) > 0:
        print('Dropping variables:', dropped_variables)
        data_df = data_df.drop(dropped_variables, axis=1)
    else:
        print('not dropping any variable')

if enable_histograms:
    print('Plotting histograms via pandas')
    data_df.hist(color='k', alpha=0.5, bins=50)
    plt.figure()
    target_sold_df.hist(color='k', alpha=0.5, bins=50)
    plt.figure()
    target_sales_df.hist(color='k', alpha=0.5, bins=50)
    plt.show()

data_df = remove_nulls(data_df, mode='zero', add_null_columns=False)

explanations = read_explanations(explanations_filename)
if verbose:
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

# Extract data for sales predictions
data_sales_sel = data[target_sold == 1.0, :]
target_sales_sel = target_sales_all[target_sold == 1.0]
print("len(data_sales_sel):", len(data_sales_sel),
      "len(target_sales_sel):", len(target_sales_sel))

# Label enlargement

############################################################################
# Data preparation for the sklearn and tensorflow algorithms

# Data splitting (train, validation, test)
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

print("len(X_m2_train):", len(X_m2_train),
      "len(X_m2_val):", len(X_m2_val),
      "len(X_m2_test):", len(X_m2_test))
if verbose:
    print("len(y_m2_train):", len(y_m2_train),
          "len(y_m2_val):", len(y_m2_val),
          "len(y_m2_test):", len(y_m2_test))


# Feature normalization (zero-mean unit-variance or range [0, 1])
# first model
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

if verbose:
    print("X_train:", X_train)
    print("X_train[0]:", X_train[0])
    print("X_train_std:", X_train_std)
print("X_train.shape:", X_train.shape)
print("X_val.shape:", X_val.shape)

X_train_std += 0.01
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# second model uses the same normalization
# X_m2_train_mean = X_m2_train.mean(axis=0)
# X_m2_train_std = X_m2_train.std(axis=0)
X_m2_train = (X_m2_train - X_train_mean) / X_train_std
X_m2_val = (X_m2_val - X_train_mean) / X_train_std
X_m2_test = (X_m2_test - X_train_mean) / X_train_std


# Removal of outliers from training data (pending)

# ML methods model 1, 2 and 3

############################################################################
# First model (M1). Estimation of whether a lead will result in 'sold'
test_acc = {}
classification_methods = []
# M1. ChanceLevel estimations
# test_acc[('ChanceLevel_MSE', 'train')] = mean_squared_error(y_train,
# y_train.mean() * np.ones_like(y_train))
# test_acc[('ChanceLevel_MSE', 'val')] = mean_squared_error(y_val,
# y_val.mean() * np.ones_like(y_val))

test_acc[('ChanceLevel_CR', 'train')] = \
    accuracy_score(y_train, stats.mode(y_train)[0] * np.ones_like(y_train))
test_acc[('ChanceLevel_CR', 'val')] = \
    accuracy_score(y_val, stats.mode(y_val)[0] * np.ones_like(y_val))

if evaluate_test_data:
    test_acc[('ChanceLevel_CR', 'test')] = \
        accuracy_score(y_test, stats.mode(y_test)[0] * np.ones_like(y_test))
    # y_test.mean() does not make sense for categorical labels

# M1. LinearRegression
if enable_linear_regression_for_sold:
    lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
    lr.fit(X_train, y_train)
    pred_lr_train = lr.predict(X_train)
    pred_lr_train = pred_lr_train.clip(0, 1)
    prob_lr_train = np.stack((1-pred_lr_train, pred_lr_train), axis=1)
    pred_lr_val = lr.predict(X_val).clip(0, 1)
    pred_lr_val = pred_lr_val.clip(0, 1)
    prob_lr_val = np.stack((1-pred_lr_val, pred_lr_val), axis=1)
    pred_lr_test = lr.predict(X_test).clip(0, 1)
    pred_lr_test = pred_lr_test.clip(0, 1)
    prob_lr_test = np.stack((1-pred_lr_test, pred_lr_test), axis=1)

    test_acc[('LinearRegression_CR', 'train')] = \
        accuracy_score(y_train, pred_lr_train >= 0.5)
    test_acc[('LinearRegression_CR', 'val')] = \
        accuracy_score(y_val, pred_lr_val >= 0.5)
    test_acc[('LinearRegression_CE', 'train')] = \
        cross_entropy(y_train, prob_lr_train)
    test_acc[('LinearRegression_CE', 'val')] = \
        cross_entropy(y_val, prob_lr_val)
    if evaluate_test_data:
        test_acc[('LinearRegression_CR', 'test')] = \
            accuracy_score(y_test, pred_lr_test >= 0.5)
        test_acc[('LinearRegression_CE', 'test')] = \
            cross_entropy(y_test, prob_lr_test)

    print(lr.coef_)
    classification_methods.append('lr')

# M1. LogisticRegression
if enable_logistic_regression:
    logr = LogisticRegression(C=1.0)
    logr.fit(X_train, y_train)
    pred_logr_train = logr.predict(X_train)
    pred_logr_val = logr.predict(X_val)
    pred_logr_test = logr.predict(X_test)
    prob_logr_train = logr.predict_proba(X_train)
    prob_logr_val = logr.predict_proba(X_val)
    prob_logr_test = logr.predict_proba(X_test)

    test_acc[('LogisticRegression_CR', 'train')] = \
        accuracy_score(y_train, pred_logr_train)
    test_acc[('LogisticRegression_CR', 'val')] = \
        accuracy_score(y_val, pred_logr_val)
    test_acc[('LogisticRegression_CE', 'train')] = \
        cross_entropy(y_train, prob_logr_train)
    test_acc[('LogisticRegression_CE', 'val')] = \
        cross_entropy(y_val, prob_logr_val)
    if evaluate_test_data:
        test_acc[('LogisticRegression_CR', 'test')] = \
            accuracy_score(y_test, pred_logr_test)
        test_acc[('LogisticRegression_CE', 'test')] = \
            cross_entropy(y_test, prob_logr_test)
    classification_methods.append('logr')

# M1. RandomForestClassifier
if enable_random_forest_classifier:
    print('Using randomized search to tune a random forest classifier...')
    # param_dist_rf = {"n_estimators": sp_randint(30, 40),
    #        "max_depth": [None, 20, 23, 25, 27, ]}
    param_dist_rf = {"n_estimators": [37]}  # zero and zero no-padding
    # param_dist_rf = {"n_estimators": [23]} # mean
    rf = RandomForestClassifier()
    rf_rs = RandomizedSearchCV(rf, n_iter=1, cv=5, n_jobs=20,
                               param_distributions=param_dist_rf)
    rf.fit(X_train, y_train)
    #
    #
    #
    rf_rs.fit(X_train, y_train)
    pred_rf_train = rf_rs.predict(X_train)
    pred_rf_val = rf_rs.predict(X_val)
    pred_rf_test = rf_rs.predict(X_test)
    prob_rf_train = rf_rs.predict_proba(X_train)
    prob_rf_val = rf_rs.predict_proba(X_val)
    prob_rf_test = rf_rs.predict_proba(X_test)

    test_acc[('RandomForestClassifier_CR', 'train')] = \
        accuracy_score(y_train, pred_rf_train)
    test_acc[('RandomForestClassifier_CR', 'val')] = \
        accuracy_score(y_val, pred_rf_val)
    test_acc[('RandomForestClassifier_CE', 'train')] = \
        cross_entropy(y_train, prob_rf_train)
    test_acc[('RandomForestClassifier_CE', 'val')] = \
        cross_entropy(y_val, prob_rf_val)
    if evaluate_test_data:
        test_acc[('RandomForestClassifier_CR', 'test')] = \
            accuracy_score(y_test, pred_rf_test)
        test_acc[('RandomForestClassifier_CE', 'test')] = \
            cross_entropy(y_test, prob_rf_test)

    print("RandomForest best_score:", rf_rs.best_score_,
          "best_params:", rf_rs.best_params_)
    classification_methods.append('rfc')

# M1. GradientBoostingClassifier
if enable_gradient_boosting_classifier:
    print('Using randomized search to tune a gradient boosting classifier...')
    # param_dist_gbc = {"n_estimators": sp_randint(95, 110),
    #                  "max_depth": sp_randint(8, 15),
    #                  "learning_rate": sp_uniform(0.17, 0.35)}
    param_dist_gbc = {"n_estimators": [102], "max_depth": [10],
                      "learning_rate": [0.24422626]}  # zero no-padding
    # param_dist_gbc = {"n_estimators": [111], "max_depth": [8],
    # "learning_rate": [0.242745]} # zero
    # param_dist_gbc = {"n_estimators": [119], "max_depth": [7],
    # "learning_rate": [0.328158]} # mean
    # GradientBoostingClassifier best_score: 0.964508496451 best_params:
    # {'n_estimators': 119, 'learning_rate': 0.32815821385475585,
    # 'max_depth': 7}
    gbc = GradientBoostingClassifier()
    gbc_rs = RandomizedSearchCV(gbc, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_gbc)
    gbc_rs.fit(X_train, y_train)
    pred_gbc_train = gbc_rs.predict(X_train)
    pred_gbc_val = gbc_rs.predict(X_val)
    pred_gbc_test = gbc_rs.predict(X_test)
    prob_gbc_train = gbc_rs.predict_proba(X_train)
    prob_gbc_val = gbc_rs.predict_proba(X_val)
    prob_gbc_test = gbc_rs.predict_proba(X_test)
    test_acc[('GradientBoostingClassifier_CR', 'train')] = \
        accuracy_score(y_train, pred_gbc_train)
    test_acc[('GradientBoostingClassifier_CR', 'val')] = \
        accuracy_score(y_val, pred_gbc_val)
    test_acc[('GradientBoostingClassifier_CE', 'train')] = \
        cross_entropy(y_train, prob_gbc_train)
    test_acc[('GradientBoostingClassifier_CE', 'val')] = \
        cross_entropy(y_val, prob_gbc_val)
    if evaluate_test_data:
        test_acc[('GradientBoostingClassifier_CR', 'test')] = \
            accuracy_score(y_test, pred_gbc_test)
        test_acc[('GradientBoostingClassifier_CE', 'test')] = \
            cross_entropy(y_test, prob_gbc_test)

    print("GradientBoostingClassifier best_score:", gbc_rs.best_score_,
          "best_params:", gbc_rs.best_params_)
    classification_methods.append('gbc')

# M1. SVC (support vector machine / classification)
if enable_support_vector_classifier:
    print('Using randomized search to tune a support vector classifier...')
    # param_dist_svc = {"C": [2.0**k for k in np.arange(2, 5, 0.1)], 
    #        'gamma': sp_uniform(0.03, 0.04)}
    param_dist_svc = {"C": [12.996], 'gamma': [0.03562]}
    # SVC best_score: 0.943213594321 best_params: {'C': 12.99603834169977,
    # 'gamma': 0.03562230562818298}
    # param_dist_svc = {"C": [32]}
    # SVC best_score: 0.941277694128 best_params: {'C': 32}, zero, no-padding
    svc = SVC(kernel='rbf', probability=True)
    svc_rs = RandomizedSearchCV(svc, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_svc)
    svc_rs.fit(X_train, y_train)
    pred_svc_train = svc_rs.predict(X_train)
    pred_svc_val = svc_rs.predict(X_val)
    pred_svc_test = svc_rs.predict(X_test)
    prob_svc_train = svc_rs.predict_proba(X_train)
    prob_svc_val = svc_rs.predict_proba(X_val)
    prob_svc_test = svc_rs.predict_proba(X_test)
    test_acc[('SVC_CR', 'train')] = \
        accuracy_score(y_train, pred_svc_train)
    test_acc[('SVC_CR', 'val')] = \
        accuracy_score(y_val, pred_svc_val)
    test_acc[('SVC_CE', 'train')] = \
        cross_entropy(y_train, prob_svc_train)
    test_acc[('SVC_CE', 'val')] = \
        cross_entropy(y_val, prob_svc_val)
    if evaluate_test_data:
        test_acc[('SVC_CR', 'test')] = \
            accuracy_score(y_test, pred_svc_test)
        test_acc[('SVC_CE', 'test')] = \
            cross_entropy(y_test, prob_svc_test)

    print("SVC best_score:", svc_rs.best_score_,
          "best_params:", svc_rs.best_params_)
    classification_methods.append('svc')

eval_batch_size = 1000
num_iter_dnn = 2
if enable_dnn_classifier:
    print('Training a DNNClassifier')
    # tf.logging.set_verbosity(tf.logging.INFO)
    my_feature_columns = []

    X_train_df = pd.DataFrame(data=X_train, columns=data_df.columns)
    X_val_df = pd.DataFrame(data=X_val, columns=data_df.columns)
    X_test_df = pd.DataFrame(data=X_test, columns=data_df.columns)
    y_train_df = pd.DataFrame(data=y_train, columns=['label_1'])
    y_val_df = pd.DataFrame(data=y_val, columns=['label_1'])
    y_test_df = pd.DataFrame(data=y_test, columns=['label_1'])

    for key in X_train_df.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # train_steps = 15000  # 20000
    param_dist_dnn = {'batch_size': [70, 75, 80, 85],
                      'hidden_0': [55, 60, 65],
                      'hidden_1': [10, 13, 15, 17],
                      'hidden_2': [10, 11, 12],
                      'dropout': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                      'batch_norm': [False],  # Not available in TF 1.8
                      'train_steps': [7500, 10000, 12500, 15000]}

    # best params: {'train_steps': 10000, 'hidden_2': 11, 'hidden_1': 13,
    # 'hidden_0': 60, 'dropout': 0.125, 'batch_size': 80, 'batch_norm': False}
    # best CE validation: 0.151061799733
    # best CR validation: 0.94730049473

    # best params: {'train_steps': 15000, 'hidden_2': 11, 'hidden_1': 15,
    # 'hidden_0': 60, 'dropout': 0.15, 'batch_size': 75, 'batch_norm': False}
    # best CE validation: 0.150819466963
    # best CR validation: 0.952247795225

    best_params = None
    best_model = None
    best_CE_val = None
    best_CR_val = None
    best_prob_dnn_train = None
    best_prob_dnn_val = None
    best_prob_dnn_test = None
    for i in range(num_iter_dnn):
        param_list = list(ParameterSampler(param_dist_dnn, n_iter=1))[0]
        print('Parameters for DNN:', param_list)
        batch_size = param_list['batch_size']
        hidden_0 = param_list['hidden_0']
        hidden_1 = param_list['hidden_1']
        hidden_2 = param_list['hidden_2']
        dropout = param_list['dropout']
        batch_norm = param_list['batch_norm']
        train_steps = param_list['train_steps']

        # Build 3 hidden layer DNN with hidden_0 -- hidden_2 units respectively
        dnn = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[hidden_0, hidden_1, hidden_2],
            # The model must choose between 2 classes.
            n_classes=2,
            # batch_norm=batch_norm,
            dropout=dropout)  # dropout probability

        # Train the Model.
        dnn.train(
            input_fn=lambda: train_input_fn(X_train_df,
                                            y_train_df, batch_size),
            steps=train_steps)

        # Evaluate the model.
        eval_result = dnn.evaluate(
            input_fn=lambda: eval_input_fn(X_val_df, y_val_df, batch_size))

        # print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))

        predictions_dnn_train = dnn.predict(
            input_fn=lambda: eval_input_fn(X_train_df, None, batch_size))
        pred_dnn_train, prob_dnn_train = \
            extract_pred_and_prob_from_estimator_predictions(predictions_dnn_train)

        predictions_dnn_val = dnn.predict(
            input_fn=lambda: eval_input_fn(X_val_df, None, eval_batch_size))
        pred_dnn_val, prob_dnn_val = \
            extract_pred_and_prob_from_estimator_predictions(predictions_dnn_val)

        predictions_dnn_test = dnn.predict(
            input_fn=lambda: eval_input_fn(X_test_df, None, eval_batch_size))
        pred_dnn_test, prob_dnn_test = \
            extract_pred_and_prob_from_estimator_predictions(predictions_dnn_test)

        CR_train = accuracy_score(y_train, pred_dnn_train)
        CR_val = accuracy_score(y_val, pred_dnn_val)
        CR_test = accuracy_score(y_test, pred_dnn_test)

        CE_train = cross_entropy(y_train, prob_dnn_train)
        CE_val = cross_entropy(y_val, prob_dnn_val)
        CE_test = cross_entropy(y_test, prob_dnn_test)

        print('CE train:', CE_train)
        print('CE validation:', CE_val)
        print('CR train:', CR_train)
        print('CR validation:', CR_val)

        # Select best model according to cross entropy (validation data)
        # TODO: Order all results by increasing CE and display for analysis
        # instead of keeping just the best CE
        if best_CE_val is None or best_CE_val > CE_val:
            best_CE_train = CE_train
            best_CE_val = CE_val
            best_CE_test = CE_test
            best_CR_train = CR_train
            best_CR_val = CR_val
            best_CR_test = CR_test
            best_params = param_list
            best_model = dnn
            best_prob_dnn_train = prob_dnn_train
            best_prob_dnn_val = prob_dnn_val
            best_prob_dnn_test = prob_dnn_test

    print('best params:', best_params)
    print('best CE validation:', best_CE_val)
    print('best CR validation:', best_CR_val)
    test_acc[('DNN_CR', 'train')] = best_CR_train
    test_acc[('DNN_CR', 'val')] = best_CR_val
    test_acc[('DNN_CE', 'train')] = best_CE_train
    test_acc[('DNN_CE', 'val')] = best_CE_val
    prob_dnn_train = best_prob_dnn_train
    prob_dnn_val = best_prob_dnn_val
    prob_dnn_test = best_prob_dnn_test

    classification_methods.append('dnn')

print(test_acc)

res_sold_df = pd.DataFrame()
for (alg, test_set) in test_acc.keys():
    res_sold_df.loc[alg, test_set] = test_acc[(alg, test_set)]
print(res_sold_df)

############################################################################
# Second model (M2). Estimation of sales for successful leads
test_m2_acc = {}
regression_methods = []
# Chance levels
test_m2_acc[('ChanceLevel_MSE', 'train')] = \
    mean_squared_error(y_m2_train, y_m2_train.mean() * np.ones_like(y_m2_train))
test_m2_acc[('ChanceLevel_MSE', 'val')] = \
    mean_squared_error(y_m2_val, y_m2_val.mean() * np.ones_like(y_m2_val))
if evaluate_test_data:
    test_m2_acc[('ChanceLevel_MSE', 'test')] = \
        mean_squared_error(y_m2_test, y_m2_test.mean() *
                           np.ones_like(y_m2_test))

# M2. LinearRegression
if enable_linear_regression_for_sales:
    lr_m2 = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
    lr_m2.fit(X_m2_train, y_m2_train)
    pred_lr_train = lr_m2.predict(X_m2_train)
    pred_lr_val = lr_m2.predict(X_m2_val)
    pred_lr_test = lr_m2.predict(X_m2_test)
    test_m2_acc[('LinearRegression_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_lr_train)
    test_m2_acc[('LinearRegression_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_lr_val)
    if evaluate_test_data:
        test_m2_acc[('LinearRegression_MSE', 'test')] = \
            mean_squared_error(y_m2_test, pred_lr_test)
    regression_methods.append('lr')

# M2. Ridge regression
if enable_ridge_regression:
    print('Using randomized search to tune ridge regression...')
    # param_dist_rid = {"alpha": sp_uniform(0.0,0.1)}
    param_dist_rid = {"alpha": [0.00608446]}  # zero no-padding
    # param_dist_rid = {"alpha": [0.0052525]} # zero
    # param_dist_rid = {"alpha": [0.006925]} # mean
    # Ridge best_score: 0.529470507097 best_params:
    # {'alpha': 0.006925342341750873}
    rid = Ridge(alpha=0.125, normalize=True)
    rid_rs = RandomizedSearchCV(rid, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_rid)
    rid_rs.fit(X_m2_train, y_m2_train)
    pred_rid_train = rid_rs.predict(X_m2_train)
    pred_rid_val = rid_rs.predict(X_m2_val)
    pred_rid_test = rid_rs.predict(X_m2_test)
    test_m2_acc[('Ridge_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_rid_train)
    test_m2_acc[('Ridge_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_rid_val)
    if evaluate_test_data:
        test_m2_acc[('Ridge_MSE', 'test')] = \
            mean_squared_error(y_m2_test, pred_rid_test)
    print("Ridge best_score:", rid_rs.best_score_,
          "best_params:", rid_rs.best_params_)
    regression_methods.append('rid')

# M2. RandomForestRegressor
if enable_random_forest_regressor:
    print('Using randomized search to tune a random forest regressor...')
    # param_dist_rfr = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
    param_dist_rfr = {"max_depth": [11]}  # zero, mean, and zero no-padding
    # RandomForestRegressor best_score: 0.824311702267 best_params: {'max_depth': 11}
    rfr = RandomForestRegressor(max_depth=1, random_state=0)
    rfr_rs = RandomizedSearchCV(rfr, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_rfr)
    rfr_rs.fit(X_m2_train, y_m2_train)
    pred_rfr_train = rfr_rs.predict(X_m2_train)
    pred_rfr_val = rfr_rs.predict(X_m2_val)
    pred_rfr_test = rfr_rs.predict(X_m2_test)
    test_m2_acc[('RandomForestRegressor_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_rfr_train)
    test_m2_acc[('RandomForestRegressor_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_rfr_val)
    if evaluate_test_data:
        test_m2_acc[('RandomForestRegressor_MSE', 'test')] = \
            mean_squared_error(y_m2_test, pred_rfr_test)
    print("RandomForestRegressor best_score:", rfr_rs.best_score_,
          "best_params:", rfr_rs.best_params_)
    regression_methods.append('rfr')

# M2. GradientBoostingRegressor
if enable_gradient_boosting_regressor:
    print('Using randomized search to tune a gradient boosting regressor...')
    param_dist_gbr = {"n_estimators": sp_randint(30, 55),
                      "max_depth": sp_randint(9, 12),
                      "learning_rate": sp_uniform(0.5, 0.7)}
    # param_dist_gbr = {"n_estimators": [132], "max_depth": [10],
    # "learning_rate": [0.6060324]} # zero
    # param_dist_gbr = {"n_estimators": [100], "max_depth": [3],
    # "learning_rate": [0.3560849]} # mean
    # GradientBoostingRegressor best_score: 0.84072146393 best_params:
    # {'n_estimators': 100, 'learning_rate': 0.35608492171567685,
    # 'max_depth': 3}
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=40, random_state=0, loss='ls')
    gbr_rs = RandomizedSearchCV(gbr, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_gbr)
    gbr_rs.fit(X_m2_train, y_m2_train)
    pred_gbr_train = gbr_rs.predict(X_m2_train)
    pred_gbr_val = gbr_rs.predict(X_m2_val)
    pred_gbr_test = gbr_rs.predict(X_m2_test)
    test_m2_acc[('GradientBoostingRegressor_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_gbr_train)
    test_m2_acc[('GradientBoostingRegressor_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_gbr_val)
    if evaluate_test_data:
        test_m2_acc[('GradientBoostingRegressor_MSE', 'test')] = \
            mean_squared_error(y_m2_test, pred_gbr_test)
    print("GradientBoostingRegressor best_score:", gbr_rs.best_score_,
          "best_params:", gbr_rs.best_params_)
    regression_methods.append('gbr')

# M2. Support Vector Regression
if enable_support_vector_regressor:
    print('Using randomized search to tune a support vector regressor...')
    # param_dist_svr = {"C": [2.0**k for k in np.arange(18, 20, 0.1)],
    #        'gamma': sp_uniform(0.19, 0.03),
    #        "epsilon": sp_uniform(1.4, 0.4)}
    # param_dist_svr = {"C": [691802], 'gamma': [0.194720],
    #        "epsilon":[1.4253616]}
    # SVR best_score: 0.736684495686 best_params: {'epsilon': 1.425361602175401,
    # 'C': 691802.1635233087, 'gamma': 0.1947206879933659}
    param_dist_svr = {"C": [2.0**18.2], "epsilon": [2.2150127]}
    # SVR best_score: 0.844215154692 best_params:
    # {'epsilon': 2.2150127431750146, 'C': 301124.3815723463}
    svr = SVR(kernel='rbf')
    svr_rs = RandomizedSearchCV(svr, n_iter=1, cv=5, n_jobs=20,
                                param_distributions=param_dist_svr)
    svr_rs.fit(X_m2_train, y_m2_train)
    pred_svr_train = svr_rs.predict(X_m2_train)
    pred_svr_val = svr_rs.predict(X_m2_val)
    pred_svr_test = svr_rs.predict(X_m2_test)
    test_m2_acc[('SVR_MSE', 'train')] = \
        mean_squared_error(y_m2_train, pred_svr_train)
    test_m2_acc[('SVR_MSE', 'val')] = \
        mean_squared_error(y_m2_val, pred_svr_val)
    if evaluate_test_data:
        test_m2_acc[('SVR_MSE', 'test')] = \
            mean_squared_error(y_m2_test, pred_svr_test)
    print("SVR best_score:", svr_rs.best_score_,
          "best_params:", svr_rs.best_params_)
    regression_methods.append('svr')

res_sales_df = pd.DataFrame()
for (alg, test_set) in test_m2_acc.keys():
    res_sales_df.loc[alg, test_set] = test_m2_acc[(alg, test_set)]
print(res_sales_df)

############################################################################
# Third model (M3). Direct computation of expected return
regression_methods_m3 = []
test_m3_acc = {}
# M3. RandomForestRegressor
if enable_random_forest_regressor_m3:
    print('Using randomized search to tune a random forest regressor (M3)...')
    # param_dist_rfr_m3 = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    # 11, 12, 13, 14, 15, 16, 18, 20, 22]}
    param_dist_rfr_m3 = {"max_depth": [15]}  # zero no-padding
    # param_dist_rfr_m3 = {"max_depth": [20]} # zero
    # param_dist_rfr_m3 = {"max_depth": [8]} # mean
    # RandomForestRegressor best_score: 0.638195402073 best_params:
    # {'max_depth': 8}
    rfr_m3 = RandomForestRegressor(max_depth=3, random_state=0)
    rfr_rs_m3 = RandomizedSearchCV(rfr_m3, n_iter=1, cv=5, n_jobs=20,
                                   param_distributions=param_dist_rfr_m3)
    rfr_rs_m3.fit(X_m3_train, y_m3_train)
    pred_rfr_train_m3 = rfr_rs_m3.predict(X_m3_train)
    pred_rfr_val_m3 = rfr_rs_m3.predict(X_m3_val)
    pred_rfr_test_m3 = rfr_rs_m3.predict(X_m3_test)
    test_m3_acc[('RandomForestRegressor_MSE', 'train')] = \
        mean_squared_error(y_m3_train, pred_rfr_train_m3)
    test_m3_acc[('RandomForestRegressor_MSE', 'val')] = \
        mean_squared_error(y_m3_val, pred_rfr_val_m3)
    if evaluate_test_data:
        test_m3_acc[('RandomForestRegressor_MSE', 'test')] = \
            mean_squared_error(y_m3_test, pred_rfr_test_m3)
    print("RandomForestRegressor best_score:", rfr_rs_m3.best_score_,
          "best_params:", rfr_rs_m3.best_params_)
    regression_methods_m3.append('rfr')

# M3. Support Vector Regression
if enable_support_vector_regressor_m3:
    print('Using randomized search to tune a support vector regressor (M3)...')
    # param_dist_svr_m3 = {"C": [2.0**k for k in np.arange(17, 18, 0.1)],
    #        'gamma': sp_uniform(0.00005, 0.0001),
    #        "epsilon": sp_uniform(0.45, 0.2)}
    param_dist_svr_m3 = {"C": [244589], "epsilon": [0.5315116], 'gamma': [0.00012]}
    # SVR best_score: 0.767954337244 best_params:
    # {'epsilon': 0.5315115872222275,
    # 'C': 244589.00053342702, 'gamma': 0.000119908712214389}
    svr_m3 = SVR(kernel='rbf')
    svr_rs_m3 = RandomizedSearchCV(svr_m3, n_iter=1, cv=5, n_jobs=20,
                                   param_distributions=param_dist_svr_m3)
    svr_rs_m3.fit(X_m3_train, y_m3_train)
    pred_svr_train_m3 = svr_rs_m3.predict(X_m3_train)
    pred_svr_val_m3 = svr_rs_m3.predict(X_m3_val)
    pred_svr_test_m3 = svr_rs_m3.predict(X_m3_test)
    test_m3_acc[('SVR_MSE', 'train')] = \
        mean_squared_error(y_m3_train, pred_svr_train_m3)
    test_m3_acc[('SVR_MSE', 'val')] = \
        mean_squared_error(y_m3_val, pred_svr_val_m3)
    if evaluate_test_data:
        test_m3_acc[('SVR_MSE', 'test')] = \
            mean_squared_error(y_m3_test, pred_svr_test_m3)
    print("SVR best_score:", svr_rs_m3.best_score_,
          "best_params:", svr_rs_m3.best_params_)
    regression_methods_m3.append('svr')

res_sales_m3_df = pd.DataFrame()
for (alg, test_set) in test_m3_acc.keys():
    print('alg:', alg, 'test_set:', test_set)
    print('test_m3_acc[(alg, test_set)]', test_m3_acc[(alg, test_set)])
    res_sales_m3_df.loc[alg, test_set] = test_m3_acc[(alg, test_set)]
print(res_sales_m3_df)

############################################################################
# Combination of models 1 and 2. This enables the computation of
# expected sales as prob of sale (by M1) * estimated sale value (by M2)
# The best 40 % of the leads is preserved and the remaining revenues are
# computed
total_revenue_train = target_sales_all[indices_train].sum()
total_revenue_val = target_sales_all[indices_val].sum()
total_revenue_test = target_sales_all[indices_test].sum()
print("total_revenue_train", total_revenue_train)
print("total_revenue_val", total_revenue_val)
if evaluate_test_data:
    print("total_revenue_test", total_revenue_test)

leads_kept_train = np.rint(0.40 * len(X_train)).astype(int)
leads_kept_val = np.rint(0.40 * len(X_val)).astype(int)
leads_kept_test = np.rint(0.40 * len(X_test)).astype(int)
print("leads_kept_train:", leads_kept_train)
print("leads_kept_val:", leads_kept_val)
print("leads_kept_test:", leads_kept_test)

for classification_method in classification_methods:
    if classification_method == 'rfc':
        prob_train = prob_rf_train[:, 1]
        prob_val = prob_rf_val[:, 1]
        prob_test = prob_rf_test[:, 1]
    elif classification_method == 'logr':
        prob_train = prob_logr_train[:, 1]
        prob_val = prob_logr_val[:, 1]
        prob_test = prob_logr_test[:, 1]
    elif classification_method == 'lr':
        prob_train = prob_lr_train[:, 1]
        prob_val = prob_lr_val[:, 1]
        prob_test = prob_lr_test[:, 1]
    elif classification_method == 'gbc':
        prob_train = prob_gbc_train[:, 1]
        prob_val = prob_gbc_val[:, 1]
        prob_test = prob_gbc_test[:, 1]
    elif classification_method == 'svc':
        prob_train = prob_svc_train[:, 1]
        prob_val = prob_svc_val[:, 1]
        prob_test = prob_svc_test[:, 1]
    elif classification_method == 'dnn':
        prob_train = prob_dnn_train[:, 1]
        prob_val = prob_dnn_val[:, 1]
        prob_test = prob_dnn_test[:, 1]
    else:
        raise ValueError('Unknown classification_method:',
                         classification_method)

    for regression_method in regression_methods:
        if regression_method == 'lr':
            pred_train = lr_m2.predict(X_train)
            pred_val = lr_m2.predict(X_val)
            pred_test = lr_m2.predict(X_test)
        elif regression_method == 'rid':
            pred_train = rid_rs.predict(X_train)
            pred_val = rid_rs.predict(X_val)
            pred_test = rid_rs.predict(X_test)
        elif regression_method == 'rfr':
            pred_train = rfr_rs.predict(X_train)
            pred_val = rfr_rs.predict(X_val)
            pred_test = rfr_rs.predict(X_test)
        elif regression_method == 'gbr':
            pred_train = gbr_rs.predict(X_train)
            pred_val = gbr_rs.predict(X_val)
            pred_test = gbr_rs.predict(X_test)
        elif regression_method == 'svr':
            pred_train = svr_rs.predict(X_train)
            pred_val = svr_rs.predict(X_val)
            pred_test = svr_rs.predict(X_test)
        else:
            raise ValueError('Unknown regression_method:', regression_method)

        expected_sales_train = prob_train * pred_train
        expected_sales_val = prob_val * pred_val
        expected_sales_test = prob_test * pred_test

        promising_sales_indices_train = np.argsort(expected_sales_train)
        promising_sales_indices_val = np.argsort(expected_sales_val)
        promising_sales_indices_test = np.argsort(expected_sales_test)

        top_promising_sales_indices_train = promising_sales_indices_train[-leads_kept_train:]
        top_promising_sales_indices_val = promising_sales_indices_val[-leads_kept_val:]
        top_promising_sales_indices_test = promising_sales_indices_test[-leads_kept_test:]
        if verbose:
            print("len(top_promising_sales_indices_train)=",
                  len(top_promising_sales_indices_train))
            print("len(top_promising_sales_indices_val)=",
                  len(top_promising_sales_indices_val))
            print("len(top_promising_sales_indices_test)=",
                  len(top_promising_sales_indices_test))

        print("classification:", classification_method, "regression:", regression_method)
        e_total_revenue_train_sel = expected_sales_train[top_promising_sales_indices_train].sum()
        e_total_revenue_val_sel = expected_sales_val[top_promising_sales_indices_val].sum()
        e_total_revenue_test_sel = expected_sales_test[top_promising_sales_indices_test].sum()
        if verbose:
            print("e_total_revenue_train_sel", e_total_revenue_train_sel)
            print("e_total_revenue_val_sel", e_total_revenue_val_sel)
            print("e_total_revenue_test_sel", e_total_revenue_test_sel)

        total_revenue_train_sel = target_sales_all[indices_train][top_promising_sales_indices_train].sum()
        total_revenue_val_sel = target_sales_all[indices_val][top_promising_sales_indices_val].sum()
        total_revenue_test_sel = target_sales_all[indices_test][top_promising_sales_indices_test].sum()
        print("total_revenue_train_sel", total_revenue_train_sel,
              '(%f)' % (100 * total_revenue_train_sel/total_revenue_train))
        print("total_revenue_val_sel", total_revenue_val_sel,
              '(%f)' % (100 * total_revenue_val_sel/total_revenue_val))
        print("total_revenue_test_sel", total_revenue_test_sel,
              '(%f)' % (100 * total_revenue_test_sel/total_revenue_test))

        cum_sum = target_sales_all[indices_val][promising_sales_indices_val[::-1]].cumsum()
        efficiency = 100 * cum_sum / total_revenue_val
        fraction = 100 * np.arange(1, len(cum_sum)+1) / len(indices_val)
        if enable_efficiency_plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # plt.title('Efficiency')
            plt.ylim(0, 100)
            plt.xlim(0, 100)
            plt.plot(fraction, efficiency)
            plt.xlabel('Percentage of leads kept')
            plt.ylabel('Percentage of sales')
            plt.grid(True)
            # plt.xticks(range(len(labels)), labels)
            # plt.yticks(range(len(labels)), labels)
            # fig.colorbar(cax)
            # fig.colorbar(cax, ticks=[0.0, 0.25, 0.5, .75, 1.0])
            plt.show()

for regression_method in regression_methods_m3:
    if regression_method == 'rfr':
        expected_sales_train_m3 = pred_rfr_train_m3
        expected_sales_val_m3 = pred_rfr_val_m3
        expected_sales_test_m3 = pred_rfr_test_m3
    elif regression_method == 'svr':
        expected_sales_train_m3 = pred_svr_train_m3
        expected_sales_val_m3 = pred_svr_val_m3
        expected_sales_test_m3 = pred_svr_test_m3
    else:
        raise ValueError('Unknown regression_method:', regression_method)

    print('regression method (M3):', regression_method)
    promising_sales_indices_train_m3 = np.argsort(expected_sales_train_m3)
    promising_sales_indices_val_m3 = np.argsort(expected_sales_val_m3)
    promising_sales_indices_test_m3 = np.argsort(expected_sales_test_m3)
    top_promising_sales_indices_train_m3 = \
        promising_sales_indices_train_m3[-leads_kept_train:]
    top_promising_sales_indices_val_m3 = \
        promising_sales_indices_val_m3[-leads_kept_val:]
    top_promising_sales_indices_test_m3 = \
        promising_sales_indices_test_m3[-leads_kept_test:]
    
    e_total_revenue_train_sel_m3 = \
        expected_sales_train_m3[top_promising_sales_indices_train_m3].sum()
    e_total_revenue_val_sel_m3 = \
        expected_sales_val_m3[top_promising_sales_indices_val_m3].sum()
    e_total_revenue_test_sel_m3 = \
        expected_sales_test_m3[top_promising_sales_indices_test_m3].sum()
    if verbose:
        print("e_total_revenue_train_sel_m3", e_total_revenue_train_sel_m3)
        print("e_total_revenue_val_sel_m3", e_total_revenue_val_sel_m3)
        print("e_total_revenue_test_sel_m3", e_total_revenue_test_sel_m3)

    total_revenue_train_sel_m3 = target_sales_all[indices_train][top_promising_sales_indices_train_m3].sum()
    total_revenue_val_sel_m3 = target_sales_all[indices_val][top_promising_sales_indices_val_m3].sum()
    total_revenue_test_sel_m3 = target_sales_all[indices_test][top_promising_sales_indices_test_m3].sum()
    print("total_revenue_train_sel_m3", total_revenue_train_sel_m3,
          '(%f)' % (100 * total_revenue_train_sel_m3 / total_revenue_train))
    print("total_revenue_val_sel_m3", total_revenue_val_sel_m3,
          '(%f)' % (100 * total_revenue_val_sel_m3 / total_revenue_val))
    print("total_revenue_test_sel_m3", total_revenue_test_sel_m3,
          '(%f)' % (100 * total_revenue_test_sel_m3/total_revenue_test))

############################################################################
# Variable importances
if verbose:
    print("Variable importances m1: ", rf_rs.best_estimator_.feature_importances_)
    print("Variable importances m2: ", rfr_rs.best_estimator_.feature_importances_)
    print("Variable importances m3: ", rfr_rs_m3.best_estimator_.feature_importances_)

var_importances = (rf_rs.best_estimator_.feature_importances_ +
                   rfr_rs.best_estimator_.feature_importances_ +
                   rfr_rs_m3.best_estimator_.feature_importances_) / 3
var_importances_pd = pd.DataFrame(var_importances,
                                  index = data_df.columns,
                                  columns=['importance']).sort_values('importance',
                                                                       ascending=False)

print("Variable importances:\n", var_importances_pd)
print("Variables ordered by importance:", var_importances_pd.index.tolist())
