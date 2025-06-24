# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
# import tensorflow as tf
from config import *

def get_prf(true_class, total_class, pred_class, nan_option = 'mean', nan_value = -1):
  '''
  Args:
    nan_option: valid values: 'mean', 'zero', 'value' ...
    nan_value: only used at the end when outputing f1 scores, if nan_option is 'value'
  '''
  pre = true_class / pred_class
  rec = true_class / total_class

  if nan_option == 'mean':
    pre_fix = np.nan_to_num(pre, nan = np.nanmean(pre))
    rec_fix = np.nan_to_num(rec, nan = np.nanmean(rec))
  else:#put to zeros
    pre_fix = np.nan_to_num(pre)
    rec_fix = np.nan_to_num(rec)

  f1 = 2/(pre_fix**(-1) + rec_fix**(-1))
  f1[(pre_fix) == 0 & (rec_fix == 0)] = 0

  if nan_option == 'value':
    f1[total_class==0] = np.nan

  return pre, rec, f1, total_class


def get_overall_accuracy(y_true, y_pred):
    """Calculates overall accuracy without using sklearn's accuracy_score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        A tuple containing the number of correct predictions and the total number of predictions.
    """

    if len(y_true.shape) == 1:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        y_true = np.eye(NUM_CLASS)[y_true].astype(int)
        y_pred = np.eye(NUM_CLASS)[y_pred].astype(int)

    if len(y_true.shape) >= 3:
        y_true = y_true.reshape(-1, NUM_CLASS)
        y_pred = y_pred.reshape(-1, NUM_CLASS)

    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total_predictions = y_true.shape[0]

    return correct_predictions, total_predictions


def get_class_wise_accuracy(y_true, y_pred, prf=False):
    """Calculates class-wise accuracy without using sklearn's accuracy_score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        prf: A boolean flag indicating whether to calculate precision, recall, and F1-score.

    Returns:
        A tuple containing the number of correct predictions per class and the total number of predictions per class.
    """

    if len(y_true.shape) >= 3:
        y_true = y_true.reshape(-1, NUM_CLASS)
        y_pred = y_pred.reshape(-1, NUM_CLASS)

    if len(y_true.shape) == 1:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        y_true = np.eye(NUM_CLASS)[y_true].astype(int)
        y_pred = np.eye(NUM_CLASS)[y_pred].astype(int)
    else:
        predicted_classes = np.argmax(y_pred, axis=1)
        y_pred = np.eye(NUM_CLASS)[predicted_classes].astype(int)

    correct_predictions = (y_true==y_pred) * (y_true == 1)
    true_per_class = np.sum(correct_predictions, axis=0)
    total_per_class = np.sum(y_true, axis=0)

    if prf:
        pred_w_class = np.argmax(y_pred, axis=1)
        pred_w_class = np.eye(NUM_CLASS)[pred_w_class].astype(int)
        pred_total = np.sum(pred_w_class, axis=0)
        return true_per_class, total_per_class, pred_total
    else:
        return true_per_class, total_per_class
