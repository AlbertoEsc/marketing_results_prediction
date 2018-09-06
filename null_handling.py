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
import numpy as np


def remove_nulls(data, mode="mean", add_null_columns=False, verbose=False):
    data = copy.deepcopy(data)
    data_null = pd.isnull(data)
    if verbose:
        print("data_null:\n", data_null)

    if mode == "none":
        substitute = np.nan
    elif mode == "mean":
        substitute = data.mean() # column-wise by default
    elif mode == "median":
        substitute = data.median()
    elif mode == "mode":
        substitute = data.mode().iloc[0]
    elif mode == 'zero':
        substitute = 0.0 * data.mean()
    else:
        raise ValueError("invalid mode:" + str(mode))

    print("Feature substitutions (%s) for null entries:\n" % mode, substitute)

    if mode != "none":
        columns = data.columns
        for column in columns:
            data.loc[data_null[column], column] = substitute[column]

    if add_null_columns:
        for column in columns:
            if data_null[column].values.any():
                data['null_'+column] = data_null[column]
    return data
