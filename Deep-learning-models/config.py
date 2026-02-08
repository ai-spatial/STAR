# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

import numpy as np

#Note: Some of the parameters are for the deep learning version (commented).
#They are for now kept as part of the config file in case some conditions
#were used in some function definitions in the code repo.

#Model choices
MODEL_CHOICE = 'DNN'
# Please refer to other code version in github for deep learning models
#Task (only classification version for RF has been tested)
MODE = 'classification'#'regression'
# MODE = 'regression'

#------------------GeoRF parameters------------------

#**************************ATTENTION NEEDED [1, total 3]**************************
#GeoRF structure parameters
#min and max depth for partitioning (actual number between min-max is auto-determined by significance testing)
MIN_DEPTH = 1
MAX_DEPTH = 4
N_JOBS = 32#parallel computing for RF
#*********************************************************************************

#Detailed ***Optional*** specifications
MIN_BRANCH_SAMPLE_SIZE = 10#minimum number of samples needed to continue partitioning
MIN_SCAN_CLASS_SAMPLE = 100#minimum number of samples needed for a class to be considered during partitioning optimization
FLEX_RATIO = 0.25#affects max size difference between two partitions in each split
FLEX_OPTION = True
# FLEX_TYPE = 'n_sample'
FLEX_TYPE = 'n_group_w_sample'#careful with threshold
# FLEX_TYPE = 'n_group'
MIN_GROUP_POS_SAMPLE_SIZE_FLEX = 10#minimum number of positive samples in a group
#For significance testing
SIGLVL = 0.01#significance level. #0.05
ES_THRD = 0.8#effect size threshold#1.4#0.01
MD_THRD = 0.001#0.005#mean_diff thrd


#------------------Training data related parameters------------------
#**************************ATTENTION NEEDED [2, total 3]**************************
#Train-val-test split
#Used as default function inputs
TRAIN_RATIO = 0.6
VAL_RATIO = 0.20#subset from training, e.g., 0.2 means 20% of training data will be set as validation
TEST_RATIO = 1 - TRAIN_RATIO
#*********************************************************************************


#------------------Spatial range related parameters------------------
#**************************ATTENTION NEEDED [3, total 3]**************************
#Spatial range parameters used for the CONUS crop classification data.
#In the code now xmin, xmax, ymin, and ymax are derived based on data in preprocessing
xmin = 0
xmax = 0
xmax_raw = 24.826344409999997
ymax_raw = 57.7696143
xmax = xmax_raw
ymax = ymax_raw
#The following is used in case the grouping is done by a grid
#X_DIM is kept as some visualization code used for testing purposes used those
X_DIM = np.array([xmax, ymax])#input img size in 2D
STEP_SIZE = 0.5#1 unit: degree
#*********************************************************************************

#The following is likely unused in the code shared (maybe in visualization or CONUS crop data preprocessing) and was used for specific visualizations
import math
X_DIM_RAW = np.array([xmax_raw, ymax_raw])#input img size in 2D
GRID_DIM = np.array([math.ceil(X_DIM[0]/STEP_SIZE), math.ceil(X_DIM[1]/STEP_SIZE)])
N_GROUPS = GRID_DIM[0] * GRID_DIM[1]

#Some of the paras are part of this specific implementation example, and can be changed.
#Two types of locations are used in this example for each data point:
#1. Its row and column ids in the original input frame (e.g., i,j in an image tile).
#2. Its grid cell's row and column ids in the grid (we overlay a grid on top of the original data to create unit groups. Each cell is a group).
GRID_COLS = [2,3]#The colunns to store the grid-based locations in X_loc (shape: Nx4). In this example the first two columns in X_loc store the original locations.
IMG_SIZE = 128

#------------------Additional parameters------------------
#Used if certain classes should be ignored in partitioning optimization or significance testing.
#Example: Imblanced binary classification where non-background pixels only accounts for a small proportion of samples.'''
#single vs multi
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


#For improved spatial contiguity of partitions
#This is only for cases where groups are based on a grid (i.e., each group is a grid cell).
# CONTIGUITY = False
# REFINE_TIMES = 0
CONTIGUITY = True
REFINE_TIMES = 3
MIN_COMPONENT_SIZE = 10

#predefined groups such as US counties
#unused here
# PREDEFINED_GROUPS = True
PREDEFINED_GROUPS = False
if PREDEFINED_GROUPS:
  CONTIGUITY = False

#Not used here: for folder naming in specific testing cases over different settings
EVAL_EXT = ''

#----------------------Deep learning version only (not impacting results but might still be mentioned somewhere in function definitions)----------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#The following will only be used for deep learning models (do not comment out)
# MODEL_CHOICE = 'DNN'
# MODEL_CHOICE = 'LSTM'
# MODEL_CHOICE = 'UNET'

#*****************DO NOT DELETE***********************
ONEHOT = True
TIME_SERIES = False
#*****************DO NOT DELETE***********************

if MODE == 'regression':
  SELECT_CLASS = None
  ONEHOT = False
  NUM_CLASS = 1



#Training related parameters - DL/default
PRETRAIN_EPOCH = 20
EPOCH_TRAIN = 20
BATCH_SIZE = 256*256
# Inference batch size to limit memory during predict()
# PREDICT_BATCH_SIZE = 1024
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0005
CLASSIFICATION_LOSS = 'categorical_crossentropy'
REGRESSION_LOSS = 'mean_squared_error'

INPUT_SIZE = 13#10*33+3#X.shape[1]#number of features
LAYER_SIZE = min(INPUT_SIZE, 10)


# '''Segmentation parameters'''
IMG_SIZE = 128
PATCH_SIZE = 128
PATCH_STEP_SIZE = 64
#
# #Time-series parameters
# TIME_SERIES = False
#
# '''Demo data paths'''
DEMO_X_PATH = 'X_example.npy'
DEMO_Y_PATH = 'y_example.npy'
DEMO_X_LOC_PATH = None
#
# '''Output directories'''
MODEL_DIR = 'result'


#Training related parameters: LSTM
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

#Training related parameters - UNet
if MODEL_CHOICE == 'UNET':
  PRETRAIN_EPOCH = 20#Stablize the model parameters before the partitioning starts
  EPOCH_TRAIN = 20#Number of epochs to train after each split (and equivalently, before the next split)
  BATCH_SIZE = 32
  LEARNING_RATE = 0.0001
  CLASSIFICATION_LOSS = 'categorical_crossentropy'
  REGRESSION_LOSS = 'mean_squared_error'








# import numpy as np
#
# '''Model choice'''
# MODEL_CHOICE = 'DNN'  # 'DNN' or 'UNet'
#
# #Task
# MODE = 'classification'#'regression'
#
# '''Partitioning parameters'''
# MIN_DEPTH = 2
# MAX_DEPTH = 6
# N_JOBS = 1
# MIN_BRANCH_SAMPLE_SIZE = 100  # minimum number of samples needed to continue partitioning
# MIN_SCAN_CLASS_SAMPLE = 100  # minimum number of samples needed for a class to be considered during partitioning optimization
# FLEX_RATIO = 0.1  # defines max size difference between two partitions in each split
# FLEX_OPTION = False
# FLEX_TYPE = 'n_group'
# MIN_GROUP_POS_SAMPLE_SIZE_FLEX = 10
#
# '''Significance testing'''
# SIGLVL = 0.01  # significance level
# ES_THRD = 0.8  # effect size threshold
# MD_THRD = 0.005  # mean_diff threshold
#
# '''Training parameters'''
# PRETRAIN_EPOCH = 40  # stabilize the model parameters before partitioning starts
# EPOCH_TRAIN = 40  # number of epochs to train after each split
# BATCH_SIZE = 64
# LEARNING_RATE = 0.0001
# CLASSIFICATION_LOSS = 'categorical_crossentropy'
# REGRESSION_LOSS = 'mean_squared_error'
# ONEHOT = True
# TRAIN_RATIO = 0.2
# VAL_RATIO = TRAIN_RATIO
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO
#
# '''Data parameters'''
# INPUT_SIZE = 13#10  # number of features
# NUM_CLASS = 2#23
# X_DIM = np.array([4096, 4096]).astype(int)  # spatial size of the entire tile
# STEP_SIZE = int(4096 / 8)  # grid cell size for grouping
# GRID_DIM = np.array([int(X_DIM[0] / STEP_SIZE), int(X_DIM[1] / STEP_SIZE)])
# GRID_COLS = [2, 3]  # columns to store grid-based locations in X_loc (shape: Nx4)
# N_GROUPS = 64  # number of grid-cell groups
# X_DIM_RAW = X_DIM
#
# #Additional parameters
# multi = False
# if multi:
#   SELECT_CLASS = np.array([1,2,3,4,5,6,7])
#   NUM_CLASS = 9
# else:
#   SELECT_CLASS = np.array([1])
#   NUM_CLASS = 2
#
# #Spatial contiguity refinement
# CONTIGUITY = False
# REFINE_TIMES = 0
# MIN_COMPONENT_SIZE = 10
#
# PREDEFINED_GROUPS = False
# if PREDEFINED_GROUPS:
#   CONTIGUITY = False
#
# EVAL_EXT = ''
#
