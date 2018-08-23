########################################################################
# Routines to simplify the use of null values in pandas                #
#                                                                      #
# Author: Alberto N. Escalante B.                                      #
# Date: 13.08.2018                                                     #
# E-mail: alberto.escalante@ini.rub.de                                 #
#                                                                      #
########################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import copy

import pandas as pd


def remove_nulls(data, mode="mean", add_null_columns=False):
    data = copy.deepcopy(data)
    data_null = pd.isnull(data)
    print("data_null:\n", data_null)

    if mode == "mean":
        substitute = data.mean() # column-wise by default
        # print("data.mean()", data.mean())
    elif mode == "median":
        substitute = data.median()
    elif mode == 'zero':
        substitute = 0.0 * data.mean()
    else:
        raise ValueError("invalid mode:" + str(mode))
    print("substitute:\n", substitute)

    columns = data.columns
    for column in columns:
        ## print("column:", column)
        ## print("data[column]", data[column])
        data.loc[data_null[column], column] = substitute[column]

    if add_null_columns:
        for column in columns:
            # Only add column with null flags if the
            # original column contains nans
            ## print("data[column].isnull()", data[column].isnull())
            ## print("data[column].isnull().values.any()", data[column].isnull().values.any())
            if data_null[column].values.any():
                ## print("good")
                data['null_'+column] = data_null[column]
    return data
