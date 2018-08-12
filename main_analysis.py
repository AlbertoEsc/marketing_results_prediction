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

import csv
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from null_handling import remove_nulls

filename = 'data/testdata.csv'
data = pd.read_csv(filename, decimal=',', sep=';')
print("data.head()\n", data.head())
print("data.describe()\n", data.describe())

explanations_filename = 'data/variable_explanations.csv'
explanations_file = open(explanations_filename)
explanations_reader = csv.reader(explanations_file, delimiter=';')
explanations = [row for row in explanations_reader]
explanations_file.close()
print(explanations)

data_null = pd.isnull(data)
print("data_null.head()\n", data_null.head())
data_mean = data.mean()
print("data_mean\n", data_mean)
data_median = data.median()
print("data_median\n", data_median)


# Handling of null values (nans)
data = remove_nulls(data, mode='mean', add_null_columns=True)
print("data.head()\n", data.head())

# Removal of outliers
data_np = data.values
print(data_np.shape)

pca = PCA(n_components=30)
pca.fit(data_np)
y = pca.transform(data_np)
print(data_np[0:1,:])
print(pca.inverse_transform(y[0:1,:]))
print('pca.explained_variance_:', pca.explained_variance_)
print('pca.components_:', pca.components_)
print('pca.components_[0]:', pca.components_[0])

lead_id = data_np[:, 0]
target_sold = data_np[:, 1]
target_sales = data_np[:, 2]
data_np = data_np[:, 3:]

data_np_sold = data_np[target_sold == 1.0, :]
target_sales_sold = target_sales[target_sold == 1.0]
print("len(data_np_sold):", len(data_np_sold), "len(target_sales_sold):", len(target_sales_sold))# Label enlargement

# Split of samples
RANDOM_STATE_TEST = 1234
RANDOM_STATE_VAL = 5678
X_train, X_test, ts_train, ts_test = \
        train_test_split(data_np, target_sold, test_size=0.20,
                         random_state=RANDOM_STATE_TEST)
X_train, X_val, ts_train, ts_val = \
        train_test_split(X_train, ts_train, test_size=0.25,
                         random_state=RANDOM_STATE_VAL)
print("len(X_train):", len(X_train), "len(X_val):", len(X_val), "len(X_test):", len(X_test))
print("len(ts_train):", len(ts_train), "len(ts_val):", len(ts_val), "len(ts_test):", len(ts_test))


# ML methods

