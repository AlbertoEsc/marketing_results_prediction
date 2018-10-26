########################################################################
# Routines that simplify the use of tensorflow estimators              #
#                                                                      #
# Author: Alberto N. Escalante B.                                      #
# Date: 19.09.2018                                                     #
# E-mail: alberto.escalante@ini.rub.de                                 #
#                                                                      #
########################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset


def extract_pred_and_prob_from_estimator_predictions(predictions):
    """From a prediction object computed by the estimator, this function
    extracts the prediction (class id) and probabilities and returns them
    as two numpy arrays. """
    pred = np.array([])
    prob = np.array([])
    for prediction in predictions:
        pred = np.append(pred, prediction['class_ids'][0])
        prob = np.append(prob, prediction['probabilities'])
    num_samples = len(pred)
    prob = prob.reshape((num_samples, -1))
    return pred, prob

def extract_pred_from_estimator_predictions(predictions):
    """From a prediction object computed by the estimator, this function
    extracts the prediction ('prediction') them
    as a numpy array. """
    # print('predictions:', predictions)
    pred = np.array([])
    for prediction in predictions:
        pred = np.append(pred, prediction['predictions'])
    num_samples = len(pred)
    pred = pred.reshape((num_samples, -1))
    return pred