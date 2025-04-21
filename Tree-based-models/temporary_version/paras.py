# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np

'''Model choices'''
MODEL_CHOICE = 'RF'
# Please refer to other code version in github for deep learning models
# MODEL_CHOICE = 'DNN'
# MODEL_CHOICE = 'LSTM'
# MODEL_CHOICE = 'UNET'

#ONEHOT used in partition_opt (only mentioned not used) and load_data
if MODEL_CHOICE == 'RF':
  ONEHOT = False
else:
  ONEHOT = True


TIME_SERIES = False
if MODEL_CHOICE == 'LSTM':
  TIME_SERIES = True

'''Problem'''
MODE = 'classification'#'regression'

'''STAR parameters'''
#min and max depth for partitioning (actual number between min-max is auto-determined by significance testing)
MIN_DEPTH = 1
MAX_DEPTH = 4#4
#parallel computing for RF
N_JOBS = 32

#detailed specifications
MIN_BRANCH_SAMPLE_SIZE = 10#minimum number of samples needed to continue partitioning
MIN_SCAN_CLASS_SAMPLE = 100#minimum number of samples needed for a class to be considered during partitioning optimization
FLEX_RATIO = 0.025#defines max size difference between two partitions in each split
FLEX_OPTION = True
# FLEX_TYPE = 'n_sample'
FLEX_TYPE = 'n_group_w_sample'#careful with threshold
# FLEX_TYPE = 'n_group'
MIN_GROUP_POS_SAMPLE_SIZE_FLEX = 10#minimum number of positive samples in a group

#significance testing
SIGLVL = 0.01#significance level. #0.05
ES_THRD = 0.8#effect size threshold#1.4#0.01
MD_THRD = 0.001#0.005#mean_diff thrd


'''Training data related parameters'''
TRAIN_RATIO = 0.40
VAL_RATIO = 0.20#extracted from training
TEST_RATIO = 1 - TRAIN_RATIO


#The following will only be used for deep learning models
'''Training related parameters - DL/default'''
PRETRAIN_EPOCH = 120
EPOCH_TRAIN = 120
BATCH_SIZE = 256*256
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0005
CLASSIFICATION_LOSS = 'categorical_crossentropy'
REGRESSION_LOSS = 'mean_squared_error'

'''Training related parameters: LSTM'''
if MODEL_CHOICE == 'LSTM':
  TIME_SERIES = True
  PRETRAIN_EPOCH = 60
  EPOCH_TRAIN = 60
  BATCH_SIZE = 256*256
  N_TIME = 33
  N_TIME_FEATURE = 10
  N_OTHER_FEATURE = 3
  LEARNING_RATE = 0.001
  CLASSIFICATION_LOSS = 'categorical_crossentropy'
  REGRESSION_LOSS = 'mean_squared_error'

'''Training related parameters - UNet'''
if MODEL_CHOICE == 'UNET':
  PRETRAIN_EPOCH = 20#Stablize the model parameters before the partitioning starts
  EPOCH_TRAIN = 20#Number of epochs to train after each split (and equivalently, before the next split)
  BATCH_SIZE = 32
  LEARNING_RATE = 0.0001
  CLASSIFICATION_LOSS = 'categorical_crossentropy'
  REGRESSION_LOSS = 'mean_squared_error'


'''Data related parameters
Some of the paras are part of this specific implementatio example, and can be changed.
Two types of locations are used in this example for each data point:
1. Its row and column ids in the original input frame (e.g., i,j in an image tile).
2. Its grid cell's row and column ids in the grid (we overlay a grid on top of the original data to create unit groups. Each cell is a group).
'''
GRID_COLS = [2,3]#The colunns to store the grid-based locations in X_loc (shape: Nx4). In this example the first two columns in X_loc store the original locations.
IMG_SIZE = 128

'''Additional:
Used if certain classes should be ignored in partitioning optimization or significance testing.
Example: Imblanced binary classification where non-background pixels only accounts for a small proportion of samples.'''


'''single vs multi'''
# multi = True
multi = False
if multi:
  #multi
  SELECT_CLASS = np.array([1,2,3,4,5,6,7])#multi-class (remove 0,9,10) and 8 (non-crop)
  NUM_CLASS = 9
  #in the data, there is only classes 0-8 (9 classes), not 0-10
  #use all
  # SELECT_CLASS = None # or np.array(range(NUM_CLASS))#select all classes
  #used in get_class_wise_stat() and get_score()
  #used locations marked by "#select_class"
else:
  #binary: one crop
  SELECT_CLASS = np.array([1])#select one class from the list
  NUM_CLASS = 2


'''update paras: temporary for RF'''
xmin = 0
xmax = 0
xmax_raw = 24.826344409999997
ymax_raw = 57.7696143
X_DIM_RAW = np.array([xmax_raw, ymax_raw])#input img size in 2D
xmax = xmax_raw
ymax = ymax_raw


#This is used in case the grouping is done by a grid
X_DIM = np.array([xmax, ymax])#input img size in 2D
# xmin, xmax: 0.0 24.826344409999997
# ymin, xmax: 0.0 57.7696143
INPUT_SIZE = 10*33+3#X.shape[1]#number of features
LAYER_SIZE = min(INPUT_SIZE, 10)
STEP_SIZE = 0.5#1 degree
import math
GRID_DIM = np.array([math.ceil(X_DIM[0]/STEP_SIZE), math.ceil(X_DIM[1]/STEP_SIZE)])
N_GROUPS = GRID_DIM[0] * GRID_DIM[1]


'''spatial contiguity'''
CONTIGUITY = False
REFINE_TIMES = 0
# CONTIGUITY = True
# REFINE_TIMES = 3

'''predefined groups such as US counties'''
# PREDEFINED_GROUPS = True
PREDEFINED_GROUPS = False
if PREDEFINED_GROUPS:
  CONTIGUITY = False

#Not used here: for folder naming in specific testing cases
EVAL_EXT = ''
