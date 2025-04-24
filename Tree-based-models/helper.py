# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
import pandas as pd
import os

from paras import *

def data_partition(X_in, X_mask):
  '''
  Currently designed for bi-partitioning
  Mask is from by generate_mask()
  '''

  X0 = X_in[X_mask==0]
  X1 = X_in[X_mask==1]

  return X0, X1

def update_branch_id(X_in_id, X_branch_id, new_branch_id):
  '''
  Update branch id if data are split or collapsed after testing
  Using strings of sequential local branch IDs as unique identifiers for each branch globally, e.g., '0011'
  add_id is in {0,1}
  '''

  X_branch_id[X_in_id] = new_branch_id

  return X_branch_id

#can be used outside in main function
def check_split_validity(y_part, MIN_BRANCH_SAMPLE_SIZE):
  '''
  Check if the split is valid using minimum sample size
  '''
  n_sample = 1
  for dim in range(max(len(y_part.shape)-1, 1)):
    n_sample *= y_part.shape[dim]
  if n_sample >= MIN_BRANCH_SAMPLE_SIZE:
    return 1
  else:
    return 0


def get_id_list(X_branch_id, X_set, branch_id, set_id):
  '''
  Get a list of IDs asscoiated with branch_id and train/val conditions
  set_id = 0 --> train
  set_id = 1 --> val
  '''
  id_list = np.where( (X_branch_id == branch_id) & (X_set == set_id) )
  # np.char.equal()

  return id_list

def get_train_val_data(X, y, X_loc, train_list, val_list):
  '''
  Get train and val data for any branch without split
  '''

  X_train = X[train_list]
  y_train = y[train_list]
  X_train_loc = X_loc[train_list]

  X_val = X[val_list]
  y_val = y[val_list]
  X_val_loc = X_loc[val_list]

  return X_train, y_train, X_train_loc, X_val, y_val, X_val_loc

def get_train_val_any(X_any, train_list, val_list):
  '''Get any subsets of X related data, e.g., train, loc, id, using the lists'''

  X_any_train = X_any[train_list]
  X_any_val = X_any[val_list]

  return X_any_train, X_any_val


def auto_hv_stack(data0, data1):
  '''
  For stacking X and y: if col dim > 1 --> vstack; otherwise, hstack
  '''
  # if len(data0.shape) > 1 :
  #   return np.vstack([data0, data1])
  # else:
  #   return np.hstack([data0, data1])

  return np.concatenate([data0, data1], axis = 0)



def get_grid_by_group(list_group_id):
  grid = np.zeros((8,8))
  for ii in range(list_group_id.shape[0]):
    value = list_group_id[ii]
    i = np.floor(value/8).astype(int)
    j = value % 8
    grid[i,j] = 1

  print(grid)


def get_slist(X_set_group, s0_group, s1_group):
  '''
  get slist containing 0,1 assignments
  '''
  slist = np.zeros(X_set_group.shape[0])
  for group_id in s1_group:
    slist[X_set_group == group_id] = 1

  s0_list = (slist==0) #keep 0,1 consistent with s0, s1
  s1_list = (slist==1)

  return s0_list, s1_list

def get_01_data(X_in, y_in, s0_list, s1_list):

  X0 = X_in[s0_list]
  X1 = X_in[s1_list]
  y0 = y_in[s0_list]
  y1 = y_in[s1_list]

  return X0, y0, X1, y1

def get_branch_data_grid_train_or_val(X, y, X_group, X_list, s0_group, s1_group):#X_loc, X_list, grid
  #X_list: train or validation list
  #variables named using train but are general for train or val
  X_train = X[X_list]
  y_train = y[X_list]

  X_train_group = X_group[X_list]

  s0_list, s1_list = get_slist(X_train_group, s0_group, s1_group)
  X0_train, y0_train, X1_train, y1_train = get_01_data(X_train, y_train, s0_list, s1_list)

  return X0_train, y0_train, X1_train, y1_train, s0_list, s1_list

def get_s_list_group_ids(s_list, group_ids):
  '''
  s_list is s0 or s1 which contains the indices of values;
  group_ids contains the corresponding group id of each index
  '''
  s_group = group_ids[s_list.astype(int)]
  return s_group

def get_gid_to_s_map(y_val_gid, s0, s1):
  '''
  s_group back to s indices for continued steps in partitioning
  '''
  n_gid = int(np.max(y_val_gid)) + 1
  gid_to_s_map = np.zeros((n_gid,))
  gid_to_s_map[y_val_gid[s0.astype(int)].astype(int)] = s0
  gid_to_s_map[y_val_gid[s1.astype(int)].astype(int)] = s1

  return gid_to_s_map

def get_branch_data_by_group(X, y, X_group, train_list, val_list, s0_group, s1_group):#, s0, s1; #from grid to group: X_loc, train_list, val_list, grid

  #train
  X0_train, y0_train, X1_train, y1_train, s0_train, s1_train = get_branch_data_grid_train_or_val(X, y,
                                                                                                 X_group, train_list,
                                                                                                 s0_group, s1_group)

  #val
  X0_val, y0_val, X1_val, y1_val, s0_val, s1_val = get_branch_data_grid_train_or_val(X, y,
                                                                                     X_group, val_list,
                                                                                     s0_group, s1_group)

  return X0_train, y0_train, X0_val, y0_val, X1_train, y1_train, X1_val, y1_val, s0_train, s1_train, s0_val, s1_val

def get_branch_X_id(X_id, train_list, val_list, s0_train, s1_train, s0_val, s1_val):#get_branch_id_grid
  X_id_train = X_id[train_list]
  X_id_val = X_id[val_list]

  X0_train_id = X_id_train[s0_train]
  X1_train_id = X_id_train[s1_train]
  X0_val_id = X_id_val[s0_val]
  X1_val_id = X_id_val[s1_val]

  return X0_train_id, X1_train_id, X0_val_id, X1_val_id

def get_branch_id_for(i,j):
  '''
  Get branch ID for the for-loop version
  '''

  branch_id = ''

  branch_max = 2**i
  branch_min = 0
  for ii in range(i):#(i+1)
    branch_mid_point = (branch_max + branch_min)/2
    if (j+1) <= branch_mid_point:
      branch_id = branch_id + '0'
      branch_max = branch_mid_point
    else:
      branch_id = branch_id + '1'
      branch_min = branch_mid_point

  return branch_id

def branch_id_to_loop_id(branch_id):
  '''
  Get the row id in a branch table, for level: len(branch_id)
  Example: '01' corresponds to the second row in level 2 (four branches in total at lv2)
  '''
  loop_id = 0
  power = 0
  for branch in branch_id[::-1]:
    branch = int(branch)
    loop_id = loop_id + branch * (2**power)
    power = power + 1

  return loop_id

def init_s_branch(n_groups):
  #potential increase in number of grid cells due to imbalance partition (e.g., 0.25 + 0.75)
  max_diviation_rate = (((0.5 + FLEX_RATIO)**2)*4)**np.floor((MAX_DEPTH-1)/2)
  max_size_needed = np.ceil(n_groups*max_diviation_rate).astype(int)

  # s_branch = pd.DataFrame(np.zeros(max_size_needed))
  s_branch = pd.DataFrame(-np.ones(max_size_needed, dtype = np.int32))
  s_branch = s_branch.rename(columns={0: ''})

  gid_list = np.empty(n_groups, dtype = np.int16)
  for i in range(n_groups):
      gid_list[i] = i#.astype(int)

  s_branch[''][:gid_list.shape[0]] = gid_list

  return s_branch, max_size_needed

def get_group_branch_dict(s_branch):
  group_branch = {}
  s_branch_fill = s_branch.fillna(-1)#, inplace = True
  for branch_id in s_branch_fill.keys().to_numpy():
    groups = s_branch_fill[branch_id].to_numpy()
    for i in range(groups.shape[0]):
      if groups[i] >= 0:
        group_branch[groups[i]] = branch_id
  return group_branch

def get_X_branch_id_by_group(X_group, s_branch, max_depth = MAX_DEPTH):
  group_branch = get_group_branch_dict(s_branch)#key:group_id, vaule:branch_id
  X_branch_id = np.empty(X_group.shape[0], dtype = np.dtype(('U', max_depth+1)))#add 1 just in case
  for group_id, branch_id in group_branch.items():
    X_branch_id[X_group == group_id] = branch_id
  return X_branch_id

def assign_branch_id(X_loc, s_branch, X_dim = X_DIM, max_depth = MAX_DEPTH, step_size = STEP_SIZE):

  test_data_grid = test_data_partition(s_branch, max_depth = max_depth)

  step_size = step_size / (2**np.floor((MAX_DEPTH-1)/2))
  X_loc_grid = np.floor(X_loc[:, 0:2]/step_size).astype(int)

  X_branch_id = test_data_grid[X_loc_grid[:,0], X_loc_grid[:,1]]

  return X_branch_id

def create_dir(model_dir = 'result', folder_name_ext = 'auto', separate_vis = False, \
                # eval_mode = False, eval_name = 'Many', first_time = False, \
                parent_dir_for_eval = None):
  '''Create directory
  Args:
    model_dir: root level model directory
    dir: sub directory to store  data partitionings (e.g., space-partitionings) and related info.
      The partitions are auto-learned and in each partition data follows the same distribution P(y|X) or function X-->y
    dir_ckpt: sub directory to store the checkpoints
    eval_mode (deprecated; inferred from parent folder input): this is for self-evaluation with many conditions resulting in many folders, where all folders are put into a parent folder
    first_time (deprecated): since there will be multiple calls to the function from different conditions (which should be put in the same parent folder),
                the parent folder shall not be recreated (by adding new counters such as 1,2,3...) each time. So it is only created in the
                first time in each evaluation.
    parent_dir_for_eval: folder created at the beginning of the evaluation (from main)
  '''

  if parent_dir_for_eval is not None:
    model_dir =  parent_dir_for_eval + '/' + model_dir

  # folder_name_ext = '_' + folder_name_ext + '_'
  model_dir = model_dir + '_' + folder_name_ext
  cnt = 0#used to create new folders if directories already exist

  if os.path.isdir(model_dir):
    while os.path.isdir(model_dir + '_' + str(cnt)):
      cnt = cnt + 1
    model_dir = model_dir + '_' + str(cnt)
    os.mkdir(model_dir)
  else:
    os.mkdir(model_dir)

  dir = model_dir + '/' + 'space_partitions'
  dir_ckpt = model_dir + '/' + 'checkpoints'

  os.mkdir(dir)
  os.mkdir(dir_ckpt)

  if separate_vis: #create a separate folder for Visualization
    dir_vis = model_dir + '/' + 'vis'
    os.mkdir(dir_vis)
    return model_dir, dir, dir_ckpt, dir_vis

  return model_dir, dir, dir_ckpt




def open_dir(model_dir = 'result'):
  '''Open an existing directory
  Args:
    model_dir: root level model directory
    dir: sub directory to store  data partitionings (e.g., space-partitionings) and related info.
      The partitions are auto-learned and in each partition data follows the same distribution P(y|X) or function X-->y
    dir_ckpt: sub directory to store the checkpoints
  '''

  if not os.path.isdir(model_dir):
    print('Error: Directory', model_dir, 'does not exist.')

  dir = model_dir + '/' + 'space_partitions'
  dir_ckpt = model_dir + '/' + 'checkpoints'

  if not os.path.isdir(dir):
    print('Error: Directory', dir, 'does not exist.')

  if not os.path.isdir(dir_ckpt):
    print('Error: Directory', dir_ckpt, 'does not exist.')

  return dir, dir_ckpt
