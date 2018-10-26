########################################################################
# Routines useful to observe variable correlations                     #
#                                                                      #
# Author: Alberto N. Escalante B.                                      #
# Date: 13.08.2018                                                     #
# E-mail: alberto.escalante@ini.rub.de                                 #
#                                                                      #
########################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
from sklearn.metrics import mean_squared_error


def analyze_correlation_matrix(sold_df, sales_df, data_df):
    all_data_df = pd.concat([sold_df, sales_df, data_df], axis=1) # , data_df

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    corr = all_data_df.corr()
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
    print("First correlations:", corr.values[0:3,0:3])
    ax1.grid(True)
    plt.title('Feature Correlations')
    labels= [str(column) for column in all_data_df.columns]
    print("labels:", labels)
    #plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    fig.colorbar(cax)
    # fig.colorbar(cax, ticks=[0.0, 0.25, 0.5, .75, 1.0])
    plt.show()


def normalize_sold_sales_data(sold, sales, data):
    sold_mean = sold.mean()
    sold_std = sold.std()
    sold = (sold - sold_mean) / sold_std
    sales_mean = sales.mean()
    sales_std = sales.std()
    sales = (sales - sales_mean) / sales_std
    data_means = data.mean(axis=0)
    data_stds = data.std(axis=0)
    data = (data - data_means) / data_stds
    data_all = np.concatenate((sold.reshape((-1,1)),
                           sales.reshape((-1,1)),
                           data), axis=1)
    return data_all


def analyze_using_PCA(sold, sales, data):
    data = normalize_sold_sales_data(sold, sales, data)

    # Basic experiment to see correlations
    print("**********************\nPCA")
    pca = PCA(n_components=30)
    pca.fit(data)
    y = pca.transform(data)
    # print(data[0:1,:])
    # print(pca.inverse_transform(y[0:1,:]))
    print('pca.explained_variance_:', pca.explained_variance_)
    print('pca.components_:', pca.components_)
    print('pca.components_[0]:', pca.components_[0])
    print('pca.components_[1]:', pca.components_[1])
    print('pca.components_[2]:', pca.components_[2])


def root_mean_squared_error(y_gt, y_pred):
    return mean_squared_error(y_gt, y_pred) ** 0.5
