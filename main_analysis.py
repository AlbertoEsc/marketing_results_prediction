########################################################################
# StepStone. Task 2.                                                   #
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
import numpy as np
import pandas

filename = "data/testdata.csv"
data = pandas.read_csv(filename, sep=';')
print(data.describe())
print(data.describe(include='all'))
