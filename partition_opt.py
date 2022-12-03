# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-05-11
# @License: MIT License

#notes: added semantic segmentation

import numpy as np
import tensorflow as tf
import pandas as pd

from paras import *

def groupby_sum(y, y_group, onehot = ONEHOT):
  y = y.astype(int)
  y_group = y_group.reshape([-1,1])
  y = np.hstack([y_group, y])

  y = pd.DataFrame(y)
  y = y.groupby([0]).sum()

  return y.index.to_numpy(), y.to_numpy()

def get_class_wise_stat(y_true, y_pred, y_group, mode = MODE, onehot = ONEHOT):

  if mode == 'classification':
    # n_sample = y_true.shape[0]

    if len(y_true.shape)==1:
      y_true = tf.one_hot(y_true, NUM_CLASS)
      y_pred = tf.one_hot(y_pred, NUM_CLASS)

    #reshape image or time-series labels
    #can handle shapes of N x m x m x k, where m is img size for semantic segmentation
    #or shapes of N x t x k, where t is the length of a sequence
    if len(y_true.shape)>=3:
      # n_dims = len(y_true.shape)
      # data_point_size = 1
      # for dim in range(1,n_dims-1):
      #   data_point_size *= y_true.shape[dim]
      n_pre = y_true.shape[0]
      y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
      y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])
      n_after = y_true.shape[0]

      data_point_size = int(n_after/n_pre)
      y_group = y_group.astype(int)
      y_group = np.repeat(y_group, data_point_size)

    stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    stat = stat.numpy()
    y_true = np.array(y_true)

    true_pred_w_class = y_true * np.expand_dims(stat, 1)

    #group by groups
    y_true_group, y_true_value = groupby_sum(y_true, y_group)
    true_pred_group, true_pred_value = groupby_sum(true_pred_w_class, y_group)

    return y_true_group, y_true_value, true_pred_group, true_pred_value

  else:
    # #reshape image or time-series labels
    # if len(y_true.shape)>=3:
    #   y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
    #   y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

    #may (or may not) need to revise for regression using scan methods!!!
    stat = tf.keras.losses.MSE(y_true, y_pred)
    stat = stat.numpy()
    stat_id, stat_value = groupby_sum(stat, y_group)

    return stat_id, stat_value

def get_c_b(y_true_value, true_pred_value):
  '''This calculates the total C and B.
  '''
  c = y_true_value - true_pred_value
  base = y_true_value

  c_tot = np.sum(c,0)
  base_tot = np.sum(base,0)

  b = np.expand_dims(c_tot,0) * (base / np.expand_dims(base_tot,0))
  b = np.nan_to_num(b)

  return c,b

def optimize_size(g, set_size, flex_ratio):
  min_size = (np.ceil(set_size * (1 - flex_ratio))).astype(int)
  max_size = (np.ceil(set_size * (1 + flex_ratio))).astype(int)

  sorted_g_score = np.sort(g)
  sorted_g_score = sorted_g_score[::-1]

  optimal_size = set_size
  for size in range(min_size, max_size):
    if sorted_g_score[size] < 0:
      optimal_size = size - 1
      break

  return optimal_size

##customized functions: if this is applied to spatial data and the user desires a more contiguous partitioning
# def get_connected_top_cells(g, g_group,
#                             min_partition_ratio = 0.2, flex = FLEX_OPTION, flex_ratio = 0.25):#grid

#   sorted_g = np.argsort(g,0)#second input might not be needed
#   sorted_g = sorted_g[::-1]
#   set_size = np.ceil(sorted_g.shape[0]/2).astype(int)

#   if flex:
#     set_size = optimize_size(g, set_size, flex_ratio)

#   s0 = sorted_g[0:set_size].astype(int)
#   s1 = sorted_g[set_size:].astype(int)
#   s0 = s0.reshape(-1)
#   s1 = s1.reshape(-1)

#   return s0_connected, s1_connected


def get_top_cells(g, flex = FLEX_OPTION, flex_ratio = 0.25):
  '''get the top half cells with largest values (return values are row ids)'''

  sorted_g = np.argsort(g,0)#second input might not be needed
  sorted_g = sorted_g[::-1]
  set_size = np.ceil(sorted_g.shape[0]/2).astype(int)

  if flex:
    set_size = optimize_size(g, set_size, flex_ratio)

  s0 = sorted_g[0:set_size]
  s1 = sorted_g[set_size:]

  return s0, s1

def get_score(y_true, y_pred, mode = MODE):

  score = None
  if mode == 'classification':
    if len(y_true.shape)==1:
      y_true = tf.one_hot(y_true, NUM_CLASS)
      y_pred = tf.one_hot(y_pred, NUM_CLASS)

    #reshape image or time-series labels
    if len(y_true.shape)>=3:
      y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
      y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

    # y_pred = to_categorical(np.argmax(arr, axis=1), 3)
    score = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    score = score.numpy()
  else:
    score = tf.keras.losses.MSE(y_true, y_pred)
    score = - score.numpy()

  return score

def get_split_score(y0_true, y0_pred, y1_true, y1_pred, mode = MODE):
  score0 = get_score(y0_true, y0_pred)
  score1 = get_score(y1_true, y1_pred)

  score = np.hstack([score0, score1])

  return score0, score1

'''Partitioning optimization.'''
def scan(y_true_value, true_pred_value, min_sample,
         flex = FLEX_OPTION):#connected = True, g_grid = None, X_dim = None, step_size = None,
  c,b = get_c_b(y_true_value, true_pred_value)

  max_iteration = 1000

  #init q
  q = np.zeros(y_true_value.shape[1])
  q_init = np.nan_to_num(c/b)
  for i in range(q_init.shape[1]):
    q_class = q_init[:,i]
    s_class, _ = get_top_cells(q_class)
    q[i] = np.sum(c[s_class,i]) / np.sum(b[s_class,i])

  # q = np.random.rand(y_true_value.shape[1])*2
  # q = np.exp(q)

  q_filter = np.sum(b,0) < min_sample
  q[q_filter == 1] = 1
  q[q == 0] = 1
  q = np.expand_dims(q,0)

  log_lr_prev = 0
  for i in range(max_iteration):#coordinate descent
    #update location
    g = c * np.log(q) + b * (1-q)
    g = np.sum(g, 1)
    s0, s1 = get_top_cells(g)
    log_lr = np.sum(g[s0])

    #update q
    q = np.nan_to_num(np.sum(c[s0],0) / np.sum(b[s0],0))
    q[q == 0] = 1
    q[q_filter == 1] = 1

    if log_lr < 0:
      print("log_lr < 0: check initialization!")

    # if (log_lr - log_lr_prev) / log_lr_prev < 0.05:
    #   break

    log_lr_prev = log_lr

    s0 = s0.reshape(-1)
    s1 = s1.reshape(-1)
    # if (i == max_iteration - 1) and (connected == True):
      # s0, s1 = get_connected_top_cells(g, g_grid, X_dim, step_size, flex = flex)

  return s0, s1
