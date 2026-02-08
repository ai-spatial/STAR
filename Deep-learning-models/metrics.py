# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

import numpy as np
import tensorflow as tf
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

  # this was for RF version (RF version is now in a separate repo with no tf content)
  # if len(y_true.shape) == 1:
  #   y_true = tf.one_hot(y_true, NUM_CLASS)
  #   y_pred = tf.one_hot(y_pred, NUM_CLASS)

  #reshape image or time-series labels
  if len(y_true.shape)>=3:
    y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
    y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

  num_class = y_true.shape[1]
  stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)

  true_pred_w_class = y_true * np.expand_dims(stat, 1)
  true = np.sum(true_pred_w_class, axis = 0).reshape(-1)
  total = np.sum(y_true, axis = 0).reshape(-1)

  # print('y_true.shape: ', y_true.shape)
  # print('true.shape: ', true.shape)
  # print('total.shape: ', total.shape)

  if prf:
    pred_w_class = tf.math.argmax(y_pred, axis = 1)
    pred_w_class = tf.one_hot(pred_w_class, depth = NUM_CLASS).numpy()
    pred_total = np.sum(pred_w_class, axis = 0).reshape(-1)
    return true, total, pred_total
  else:
    return true, total


'''Define some additional losses (e.g., for class imbalance issues)'''

# def convert_to_flat_tensor(y_true, y_pred, mode = MODE):
#   if mode == 'classification':
#     if len(y_true.shape)==1:
#       y_true = tf.one_hot(y_true, NUM_CLASS)
#       y_pred = tf.one_hot(y_pred, NUM_CLASS)
#     else:
#     #this is to make coding consistent for later parts of the function (where tf functions are used)
#       y_true = tf.convert_to_tensor(y_true)
#       y_pred = tf.convert_to_tensor(y_pred)
#       # tf.convert_to_tensor(numpy_array, dtype=tf.float32)
#
#     #reshape image or time-series labels
#     if len(y_true.shape)>=3:
#       y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
#       y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])
#
#     if SELECT_CLASS is not None and not MULTI:
#       y_true = y_true[:, SELECT_CLASS[0]]
#       y_pred = y_pred[:, SELECT_CLASS[0]]
#
#     return y_true, y_pred

def convert_to_flat_tensor(y_true, mode = MODE):
  if mode == 'classification':

    if tf.is_tensor(y_true):#isinstance(y_true,np.ndarray):
      if y_true.dtype != tf.float32:
        y_true = tf.cast(y_true, tf.float32)

    if len(y_true.shape)==1:
      y_true = tf.one_hot(y_true, NUM_CLASS)
    else:
    #this is to make coding consistent for later parts of the function (where tf functions are used)
      y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
      # tf.convert_to_tensor(numpy_array, dtype=tf.float32)

    #reshape image or time-series labels
    if len(y_true.shape)>=3:
      y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays

    if SELECT_CLASS is not None and not multi:
      y_true = y_true[:, SELECT_CLASS[0]]

    return y_true

def dice_coef(y_true, y_pred, sample_weights = None):
    smooth = 0.001
    # y_true = y_true[:,:,:,1]
    # y_pred = tf.keras.activations.relu(y_pred[:,:,:,1], 0.5)
    # y_true, y_pred = convert_to_flat_tensor(y_true, y_pred)
    y_true = convert_to_flat_tensor(y_true)
    y_pred = convert_to_flat_tensor(y_pred)

    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred, sample_weights = None):
    # GT and predicted have shape = [batch_size, d0, .. dN]
    return 1-dice_coef(y_true, y_pred)

def weighted_cross_entropy_loss(y_true, y_pred, sample_weights = None):
    s = y_true.shape

    # y_true = y_true[:,:,:,0]
    # y_pred = y_pred[:,:,:,0]
    y_true = convert_to_flat_tensor(y_true)
    y_pred = convert_to_flat_tensor(y_pred)

    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    # if s[3]>=2:
    if sample_weights is not None:
        # weight = tf.cast(y_true[:,:,:,1], tf.float32)
        weight = tf.cast(sample_weights, tf.float32)
        weight_f = tf.keras.layers.Flatten()(weight)
    else:
        print('error! sample weight is not passed!')
        # weight_f = tf.cast(tf.where(y_true_f==1, 1., 1.), tf.float32)
        weight_f = tf.cast(tf.where(y_true_f==1, 5., 1.), tf.float32)

    # # weight = np.ones(y_true_f.shape)
    # # weight = tf.convert_to_tensor(weight, dtype=tf.float32)
    # if s[3]==1:
    #     weight_f = tf.cast(tf.where(y_true_f==1, 1., 1.), tf.float32)

    eps = 1e-7
    log_loss_weight = -1.*tf.reduce_mean(weight_f*(y_true_f * tf.math.log(y_pred_f + eps) +
                                  (1-y_true_f) * tf.math.log(1-y_pred_f + eps)))

    return log_loss_weight

def mixed_dice_cross_entropy_loss(y_true, y_pred, sample_weights = None):#
    weight_dice = 0.5
    weight_ce = 1 - weight_dice

    return weight_dice*dice_coef_loss(y_true, y_pred) + \
           weight_ce*weighted_cross_entropy_loss(y_true, y_pred, sample_weights)
