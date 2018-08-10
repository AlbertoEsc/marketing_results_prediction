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

    data['Target_Sales']
    data_null['Target_Sales']

    if mode == "mean":
        substitute = data.mean()
    elif mode == "median":
        substitute = data.median()
    else:
        raise ValueError("invalid mode:" + str(mode))
    print("substitute:\n", substitute)
    substitute['Target_Sales']

    columns = data.columns
    for column in columns:
        print("column:", column)
        print("data[column]", data[column])
        data.loc[data_null[column], column] = substitute[column]

    if add_null_columns:
        for column in columns:
            data['null_'+column] = data_null[column]
    return data
