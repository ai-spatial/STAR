# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np

'''STAR parameters'''
#Determine the min and max depths for the partitioning hierarchy
MIN_DEPTH = 2
MAX_DEPTH = 6
#Partitioning optimization
MIN_BRANCH_SAMPLE_SIZE = 100#minimum number of samples needed to continue partitioning
MIN_SCAN_CLASS_SAMPLE = 100#minimum number of samples needed for a class to be considered during partitioning optimization
FLEX_RATIO = 0.1#defines max size difference between two partitions in each split
FLEX_OPTION = False
#significance testing
SIGLVL = 0.01#significance level. #0.05
ES_THRD = 0.8#effect size threshold#1.4#0.01
MD_THRD = 0.005#mean_diff thrd


# '''Training related parameters'''
# PRETRAIN_EPOCH = 60#Stablize the model parameters before the partitioning starts
# EPOCH_TRAIN = 60#Number of epochs to train after each split (and equivalently, before the next split)
# BATCH_SIZE = 256*256
# LEARNING_RATE = 0.001
# MODE = 'classification'#'regression'
# CLASSIFICATION_LOSS = 'categorical_crossentropy'
# REGRESSION_LOSS = 'mean_squared_error'
# ONEHOT = True
# TRAIN_RATIO = 0.125
# VAL_RATIO = TRAIN_RATIO
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO


'''Training related parameters - UNet'''
PRETRAIN_EPOCH = 40#Stablize the model parameters before the partitioning starts
EPOCH_TRAIN = 40#Number of epochs to train after each split (and equivalently, before the next split)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
MODE = 'classification'#'regression'
CLASSIFICATION_LOSS = 'categorical_crossentropy'
REGRESSION_LOSS = 'mean_squared_error'
ONEHOT = True
TRAIN_RATIO = 0.2
VAL_RATIO = TRAIN_RATIO
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO


'''Data related parameters
Some of the paras are part of this specific implementatio example, and can be changed.
Two types of locations are used in this example for each data point:
1. Its row and column ids in the original input frame (e.g., i,j in an image tile).
2. Its grid cell's row and column ids in the grid (we overlay a grid on top of the original data to create unit groups. Each cell is a group).
'''
INPUT_SIZE = 10#number of features. can be an image (may need some updates). tested on time-series inputs
NUM_CLASS = 23
X_DIM = np.array([4096,4096]).astype(int)#spatial size of the entire satellite imagery tile. The input is not necessarily an image tile, just an example.
STEP_SIZE = int(4096 / 8)#We overlay a grid on top of the image tile. This is the cell size of each grid cell. The grid is used to create groups.
GRID_DIM = np.array([int(X_DIM[0]/STEP_SIZE), int(X_DIM[1]/STEP_SIZE)])
GRID_COLS = [2,3]#The colunns to store the grid-based locations in X_loc (shape: Nx4). In this example the first two columns in X_loc store the original locations.
N_GROUPS = 64#The grid cells are the groups.
#UNet paras
IMG_SIZE = 128
