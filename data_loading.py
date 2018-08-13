from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import csv
import pandas as pd

def read_data(filename):
    data = pd.read_csv(filename, decimal=',', sep=';')
    # print("data.head()\n", data.head())
    # print("data.describe()\n", data.describe())

    # data_null = pd.isnull(data)
    # print("data_null.head()\n", data_null.head())
    # data_mean = data.mean()
    # print("data_mean\n", data_mean)
    # data_median = data.median()
    # print("data_median\n", data_median)

    # Handling of null values (nans)

    # Why was this weird column name used? '\xef\xbb\xbfID'
    id_df = data['ID'] # ID
    target_sold_df = data['Target_Sold'] # ).copy()
    target_sales_df = data['Target_Sales'] # ).copy()
    data = data.drop(['ID', 'Target_Sold', 'Target_Sales'], axis=1)
    # print("data.head()\n", data.head())
    return data, id_df, target_sold_df, target_sales_df


def read_explanations(explanations_filename):
    explanations_file = open(explanations_filename)
    explanations_reader = csv.reader(explanations_file, delimiter=';')
    explanations = [row for row in explanations_reader]
    explanations_file.close()
    # print(explanations)
    return explanations