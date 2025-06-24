# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
# import tensorflow as tf
import pandas as pd

# from data_syn_exp import get_swap_group_list

from config import *

def load_data():
    # real data
    X = np.load('/content/drive/MyDrive/Spatialnet/Sentinel_2018_T11SKA_data/feat_T11SKA_20m.npy')
    y = np.load('/content/drive/MyDrive/Spatialnet/Sentinel_2018_T11SKA_data/label_T11SKA_20m.npy').astype(int)
    if ONEHOT:
        y = np.reshape(y,[-1])
        # y = tf.one_hot(y, NUM_CLASS).numpy()
        y = np.eye(NUM_CLASS)[y].astype(int)
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


def load_demo_data(full = False, onehot = ONEHOT):
  '''
  Load demo dataset for CONUS crop classification.
  Data points: This is a subset of all data points (10% random samples) for the full dataset to reduce the size.
  Features: For X, the full data has 333 features per data point, including 10 band values over 33 time steps + 3 topographical features.
      The demo data has 13 features (10 band values from August + 3 topographical features).
  Labels: Same set of 10% randomly sampled data points. Label is binary.
  '''
  CROP_CHOICE = crop_type#used earlier
  #soybean, corn, wheat, cotton

  if full:
    X = np.load('X_full.npy')#full data is very large in size
    y = np.load('y.npy')
    X_loc = np.load('X_loc.npy')
  else:
    X = np.load('X_demo.npy')
    y = np.load('y_demo.npy')
    X_loc = np.load('X_loc_demo.npy')

  if onehot:
    y = y_to_onehot(y)

  return X, y, X_loc

def load_data_us_cdl(full = False, from_raw = True, crop_type = 'corn', onehot = ONEHOT):
  # file_name = '/content/drive/MyDrive/CDLTrain2021/all.tiles.replaced.csv'

  CROP_CHOICE = crop_type#used earlier
  #soybean, corn, wheat, cotton

  if full:
    X = np.load('X_full.npy')
  else:
    X = np.load('X.npy')

  # y = np.load('y.npy')
  y = np.load('y_' + CROP_CHOICE + '.npy')
  X_loc = np.load('X_loc.npy')


  if onehot:
    y = y_to_onehot(y)

  return X, y, X_loc

def y_to_onehot(y):
  y = np.reshape(y,[-1])
  # y = tf.one_hot(y, NUM_CLASS).numpy()
  y = np.eye(NUM_CLASS)[y].astype(int)
  y = y.astype(int)
  return y

def merge_labels(y, y_map):
  '''
  Args:
    y: labels of data samples
    y_map: two columns are original label IDs and new label IDs
  '''

  y_new = np.copy(y)#make a copy (mutable objects)

  for i in range(y_map.shape[0]):
    y_new[y==y_map[i,0]] = y_map[i,1]

  return y_new

def project_X_loc(X_loc):
  '''Make X_loc min values to 0.'''
  xmin = np.min(X_loc[:,0])
  xmax = np.max(X_loc[:,0])
  ymin = np.min(X_loc[:,1])
  ymax = np.max(X_loc[:,1])
  print('before:')
  print('xmin, xmax:', np.min(X_loc[:,0]), np.max(X_loc[:,0]))
  print('ymin, xmax:', np.min(X_loc[:,1]), np.max(X_loc[:,1]))

  X_loc[:,0] = X_loc[:,0] - xmin
  X_loc[:,1] = X_loc[:,1] - ymin

  xmin = np.min(X_loc[:,0])
  xmax = np.max(X_loc[:,0])
  ymin = np.min(X_loc[:,1])
  ymax = np.max(X_loc[:,1])
  print('after:')
  print('xmin, xmax:', np.min(X_loc[:,0]), np.max(X_loc[:,0]))
  print('ymin, xmax:', np.min(X_loc[:,1]), np.max(X_loc[:,1]))

  return X_loc, xmin, xmax, ymin, ymax

def rev_project_X_loc(X_loc, xmin, ymin):
  '''Revert the projection.'''
  X_loc[:,0] = X_loc[:,0] + xmin
  X_loc[:,1] = X_loc[:,1] + ymin

  xmin = np.min(X_loc[:,0])
  xmax = np.max(X_loc[:,0])
  ymin = np.min(X_loc[:,1])
  ymax = np.max(X_loc[:,1])
  print('xmin, xmax:', np.min(X_loc[:,0]), np.max(X_loc[:,0]))
  print('ymin, xmax:', np.min(X_loc[:,1]), np.max(X_loc[:,1]))

  return X_loc

def reload_X_loc_raw():
  X_loc = np.load('X_loc.npy')
  return X_loc

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
