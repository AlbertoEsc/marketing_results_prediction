########################################################################
# Routines to load the data from the csv files and to process it       #
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


def read_data(filename):
    """Reads the data from a csv file"""
    data = pd.read_csv(filename, decimal=',', sep=';')
    # print("data.head()\n", data.head())
    # print("data.describe()\n", data.describe())

    # Why was this weird column name used? '\xef\xbb\xbfID'
    id_df = data['Index'] # I
    target_sold_df = data['Contract_Closed'] # ).copy()
    target_sales_df = data['Contract_Value'] # ).copy()
    data = data.drop(['Index', 'Contract_Closed', 'Contract_Value'], axis=1)
    #print(list(data.columns))
    #quit()
    if 'Unnamed: 0' in list(data.columns):
        data = data.drop(['Unnamed: 0'], axis=1)
        #quit()
    return data, id_df, target_sold_df, target_sales_df


def read_explanations(explanations_filename):
    """Reads the variable explanations from a csv file"""
    explanations_file = open(explanations_filename)
    explanations_reader = csv.reader(explanations_file, delimiter=';')
    explanations = [row for row in explanations_reader]
    explanations_file.close()
    return explanations


def compute_indices_train_val_test(num_samples, frac_val=0.2, frac_test=0.2):
    """Computes sample indices that divide the data in
    training/validation/test sets.

    The proportion of samples used for the validation and test sets is given
    by two non-negative float numbers frac_val and frac_test (the remaining
    samples are the training data)."""
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)
    samples_train = int(num_samples * (1.0 - frac_val - frac_test))
    samples_val = int(num_samples * frac_val)
    indices_train = all_indices[0:samples_train]
    indices_val = all_indices[samples_train:samples_train + samples_val]
    indices_test = all_indices[samples_train + samples_val:num_samples]
    return indices_train, indices_val, indices_test


def split_data(data, indices_train, indices_val, indices_test):
    """Divides the ndarray data into three ndarrays, by using the provided
    indices."""
    return data[indices_train], data[indices_val], data[indices_test]