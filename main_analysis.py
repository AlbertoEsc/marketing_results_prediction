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


data = remove_nulls(data, mode='mean', add_null_columns=True)
print("data.head()\n", data.head())