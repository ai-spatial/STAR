# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from config import *
# from customize import generate_groups, generate_groups_from_raw_loc#can customize group definition
from customize import *#generate_groups

#data preprocessing. Defined in config.py
# TRAIN_RATIO = 0.25
# VAL_RATIO = TRAIN_RATIO
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

#consider moving generation of X_set, X_id, X_branch_id, etc. into this function itself
#or as a separate function

def init_X_info(X, y, X_loc = None):
  '''Initialize all X related info needed for training.'''

  if X_loc is None:
    X_loc = generate_X_loc(X_DIM).astype(int)
    X_loc = np.hstack([X_loc, generate_X_loc_grid(X_DIM, STEP_SIZE)]).astype(int)
    X_group = generate_groups(X_loc)
  else:
    X_group = generate_groups_from_raw_loc(X_loc = X_loc)

  X_set = train_val_test_split(X, y)
  X_id = np.arange(X.shape[0])#the id is used to later refer back to the original X, and the related information
  X_branch_id= init_X_branch_id(X)#stores branch_id of each data point during training (data points will be partitioned into different branches)

  return X_group, X_set, X_id, X_branch_id

def generate_X_loc(X_dim):
  '''Used when loc is not known as input.
  This function uses row and column ids of pixels in an image as locations.

  Args:
    X_dim = [n_rows, n_columns], specifying the shape of the image
  '''

  i_loc = np.zeros(X_dim)
  j_loc = np.zeros(X_dim)
  for i in range(X_dim[0]):
    i_loc[i,:] = i
  for j in range(X_dim[1]):
    j_loc[:,j] = j

  X_loc = np.hstack([i_loc.reshape((-1,1)), j_loc.reshape((-1,1))])

  return X_loc

def init_X_branch_id(X, max_depth = MAX_DEPTH):
  X_branch_id = np.empty(X.shape[0], dtype = np.dtype(('U', max_depth+1)))#add 1 just in case
  return X_branch_id

def train_val_test_split(X, y, train_ratio = TRAIN_RATIO):#X_loc,
  train_step = np.round(1/train_ratio).astype(int)
  X_set = np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    if i % train_step == 0:
      X_set[i] = 0
    elif i % train_step == 1:
      X_set[i] = 1
    else:
      X_set[i] = 2
  return X_set


def train_val_split(X, val_ratio = VAL_RATIO):
  '''This assumes data contains only train and val'''
  np.random.seed(0)
  rand_split = np.random.rand(X.shape[0])
  X_set = np.zeros(X.shape[0])
  X_set[(rand_split<val_ratio)] = 1
  return X_set


def train_test_split_all(X, y, X_loc, X_group, test_ratio = TEST_RATIO, random_state = 0):
  '''Same role as regular train-test split function but with more inputs: X_loc and X_group.'''
  np.random.seed(random_state)
  rand_split = np.random.rand(X.shape[0])
  X_set = np.zeros(X.shape[0])
  test_thrd = test_ratio
  X_set[(rand_split<test_thrd)] = 2

  Xtrain = X[X_set==0]
  Xtest = X[X_set==2]

  ytrain = y[X_set==0]
  ytest = y[X_set==2]

  Xtrain_loc = X_loc[X_set==0]
  Xtest_loc = X_loc[X_set==2]

  Xtrain_group = X_group[X_set==0]
  Xtest_group = X_group[X_set==2]

  return Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group


def generate_X_loc_grid(X_dim, step_size):
  '''Used when loc is not known as input.
  This function uses row and column ids of the grid cells that pixels belong to in an image as locations.
  Basically, it applies a grid on top of the original image, and uses a grid cell's row and column ids as the location of all pixels inside it.

  Args:
    X_dim = [n_rows, n_columns], specifying the shape of the image
    step_size: used to define the grid, i.e., cell size.
  '''
  # X_loc_dim = np.hstack([X_dim, 2])
  i_loc = np.zeros(X_dim)
  j_loc = np.zeros(X_dim)
  cnt = 0
  for i in range(0, X_dim[0], step_size):
    i1 = min(i+step_size, X_dim[0])
    i_loc[i:i1,:] = cnt
    cnt = cnt + 1

  cnt = 0
  for j in range(0, X_dim[1], step_size):
    j1 = min(j+step_size, X_dim[1])
    j_loc[:,j:j1] = cnt
    cnt = cnt + 1

  X_loc = np.hstack([i_loc.reshape((-1,1)), j_loc.reshape((-1,1))])

  return X_loc
#
# def generate_groups(X_loc):
#   '''Create groups. This is needed for data partitioning. In the example, we used each grid cell as a group.
#   Users can customize their own groups for spatial or non-spatail data by modifying this function.
#   '''
#   n_rows = np.max(X_loc[:,2])+1
#   n_cols = np.max(X_loc[:,3])+1
#   X_group = X_loc[:,2]*n_cols + X_loc[:,3]
#   return X_group

def generate_grid(X_dim, step_size, s0, s1):#, max_size_decay
  '''This creates a grid, and marks cells belonging to partition-1 as 1, partition-0 as 0.
  Note that s0 and s1 may not cover the whole grid. They only cover the whole grid when the partitioning is carried out on the root partition (entire space).

  Args:
    s0: selected grid cells (top scoring ones) for partition-0. Here each element in s0 is a tuple (row, col).
    s1: selected grid cells (top scoring ones) for partition-1. Here each element in s1 is a tuple (row, col).
  '''

  if (X_dim[0] % step_size > 0) or (X_dim[1] % step_size > 0):
    print("Warning: X_dim is not divisible by step_size: [%d, %d], %d" % (X_dim[0], X_dim[1], step_size))

  n_row = np.ceil(X_dim[0]/step_size).astype(int)
  n_col = np.ceil(X_dim[1]/step_size).astype(int)
  #empty cells will be marked as -1
  grid = -np.ones([n_row, n_col])

  rows, cols = zip(*s0)
  grid[rows, cols] = 0

  rows, cols = zip(*s1)
  grid[rows, cols] = 1

  return grid


def update_grid(X_loc, update_id_list, step_size, grid_cols = GRID_COLS):#X_dim
  '''
  grid_cols: col ids used to store grid i,j; e.g., [2,3]
  '''

  X_loc[np.ix_(update_id_list[0], grid_cols)] = np.floor(X_loc[np.ix_(update_id_list[0], [0,1])]/step_size).astype(int)

  return X_loc
