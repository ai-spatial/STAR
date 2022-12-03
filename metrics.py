# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
import tensorflow as tf
from paras import *

def get_prf(true_class, total_class, pred_class):
  pre = true_class / pred_class
  rec = true_class / total_class

  pre_fix = np.nan_to_num(pre, nan = np.nanmean(pre))
  rec_fix = np.nan_to_num(rec, nan = np.nanmean(rec))
  f1 = 2/(pre_fix**(-1) + rec_fix**(-1))
  return pre, rec, f1, total_class

def get_overall_accuracy(y_true, y_pred):

  if len(y_true.shape) == 1:
    y_true = tf.one_hot(y_true, NUM_CLASS)
    y_pred = tf.one_hot(y_pred, NUM_CLASS)

  #reshape image or time-series labels
  if len(y_true.shape)>=3:
    y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
    y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

  stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
  true = np.sum(stat)
  total = stat.shape[0]

  return true, total

def get_class_wise_accuracy(y_true, y_pred, prf = False):

  if len(y_true.shape) == 1:
    y_true = tf.one_hot(y_true, NUM_CLASS)
    y_pred = tf.one_hot(y_pred, NUM_CLASS)

  #reshape image or time-series labels
  if len(y_true.shape)>=3:
    y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
    y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

  num_class = y_true.shape[1]
  stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)

  true_pred_w_class = y_true * np.expand_dims(stat, 1)
  true = np.sum(true_pred_w_class, axis = 0).reshape(-1)
  total = np.sum(y_true, axis = 0).reshape(-1)

  if prf:
    pred_w_class = tf.math.argmax(y_pred, axis = 1)
    pred_w_class = tf.one_hot(pred_w_class, depth = NUM_CLASS).numpy()
    pred_total = np.sum(pred_w_class, axis = 0).reshape(-1)
    return true, total, pred_total
  else:
    return true, total
