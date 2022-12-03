# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from paras import *

def load_data(X_path = 'X_example.npy',
              y_path = 'y_example.npy'):
    # real data
    X = np.load(X_path)
    y = np.load(y_path).astype(int)
    if ONEHOT:
        y = np.reshape(y,[-1])
        y = tf.one_hot(y, NUM_CLASS).numpy()
    else:
        y = label_raw
        y = np.reshape(y,[-1])

    y = y.astype(int)
    X = np.reshape(X,[-1,INPUT_SIZE])

    return X, y

def img_to_patch(X, y, size = 128, step_size = 64, return_loc = False):
  '''
  Args:
    size: image size
    step_size: controls sample rate along each dimension
    return_loc: if True, returns a location value for each data point in X_loc;
      otherwise, returns only generated X and y (as image patches for semantic segmentation)
  '''
  num_row, num_col, n_features = X.shape
  row_steps = (np.floor(num_row/step_size)-1).astype(int)
  col_steps = (np.floor(num_col/step_size)-1).astype(int)
  n_patch = row_steps * col_steps
  n_class = y.shape[2]

  X_new = np.zeros((n_patch, size, size, n_features))
  y_new = np.zeros((n_patch, size, size, n_class), dtype=np.int8)

  cnt = 0
  for i in range(row_steps):
    for j in range(col_steps):
      i0 = i*step_size
      i1 = i0 + size
      j0 = j*step_size
      j1 = j0 + size
      X_new[cnt, ...] = X[i0:i1, j0:j1, :]
      y_new[cnt, ...] = y[i0:i1, j0:j1, :]
      cnt += 1

  print('n_patch:', n_patch, ', cnt:', cnt)

  if return_loc:
    X_loc = np.zeros((n_patch, 2))
    cnt = 0
    for i in range(row_steps):
      for j in range(col_steps):
        X_loc[cnt] = [i*step_size, j*step_size]
        cnt += 1
    return X_new, y_new, X_loc
  else:
    return X_new, y_new

def load_data_seg(X_path = 'X_example.npy',
                  y_path = 'y_example.npy',
                  return_loc = False):
    # real data
    X = np.load(X_path)
    y = np.load(y_path).astype(int)

    if ONEHOT:
        # y = np.reshape(y,[-1])
        y = tf.one_hot(y, NUM_CLASS).numpy()
    else:
        y = label_raw
        y = np.reshape(y,[-1])

    y = y.astype(np.int8)
    # X = np.reshape(X,[-1,INPUT_SIZE])
    # X, y = img_to_patch(X, y)
    X, y, X_loc = img_to_patch(X, y, return_loc = return_loc)

    if return_loc:
      return X, y, X_loc
    else:
      return X,y



#Not used in this version. Replaced by static_table.py
def load_sig_test_lookup_table():
    '''Load csv containing the look-up table for critical values.'''
    cvtable = np.zeros([34,7])
    import csv
    k=0
    with open('./test_dist.csv', newline='') as csvfile:
        crd = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in crd:
            cvtable[k,:] = np.array(row[:7]).astype(float)
            k+=1
    return cv_table
