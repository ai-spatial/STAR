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
from scipy import stats

# from partition_opt import swap_partition_general
from customize import generate_groups_loc
# from visualization import vis_partition_group, generate_vis_image_from_grid
from visualization import *
from helper import *

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
    else:
    #this is to make coding consistent (tf functions might be used in this function when implementing the RF version)
      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)
      # tf.convert_to_tensor(numpy_array, dtype=tf.float32)

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

    #select_class (should not be used before categorical_accuracy,
    #which is not correct when there is only one class)
    #here only use aggregated c and b, so only need to select subset of columns
    if SELECT_CLASS is not None:
      y_true = y_true[:, SELECT_CLASS]

    '''Check what happens if y_true is all zeros (already entered the optimization phase)'''

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

def get_min_max_size(cnt, flex_ratio):
  #this can be optimized with a binary search
  #since the total time cost on this should be minimal, using a linear scan at the moment
  cnt_total = np.sum(cnt)
  # print('cnt.shape, cnt_total: ', cnt.shape, cnt_total)
  cnt_to_i = 0
  min_size = -1
  max_size = -1
  for i in range(cnt.shape[0]):
    cnt_to_i += cnt[i]
    if min_size == -1 and cnt_to_i / cnt_total >= flex_ratio:
      min_size = i
    if max_size == -1 and cnt_to_i / cnt_total >= 1 - flex_ratio:
      max_size = i-1
      break

  if max_size < min_size:
    print('max_size < min_size: ', min_size, max_size)
    max_size = min_size

  return min_size, max_size


def optimize_size(g, set_size, flex_ratio, flex_type = FLEX_TYPE, cnt = None):

  sorted_g_score = np.sort(g)
  sorted_g_score = sorted_g_score[::-1]

  optimal_size = set_size

  if flex_type == 'n_sample' and cnt is not None:
    min_size, max_size = get_min_max_size(cnt.astype(float), flex_ratio)
  elif flex_type == 'n_group_w_sample' and cnt is not None:
    cnt_binary = (cnt > MIN_GROUP_POS_SAMPLE_SIZE_FLEX).astype(float)
    min_size, max_size = get_min_max_size(cnt_binary, flex_ratio)
  else:
    min_size = (np.ceil(set_size * (1 - flex_ratio))).astype(int)
    max_size = (np.ceil(set_size * (1 + flex_ratio))).astype(int)
    if flex_type != 'n_group':
      print('Warning: cnt is None')

  for size in range(min_size, max_size):
    if sorted_g_score[size] <= 0:
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

def swap_partition_general(loc_grid, locs, null_value = -1):
  '''Used as a sub-function to improve contiguity.
  locs is an arrary of locations in partition pid (pid is either 0 or 1)
  '''
  i_max, j_max = loc_grid.shape
  loc_grid_new = np.copy(loc_grid)
  for loc in locs:
    loc_i = loc[0]
    loc_j = loc[1]

    if loc_grid[loc_i, loc_j] == null_value:#confirm how many null values are there in refinement-all
      continue

    ##for 8-neighbor
    loc_i_min = max(0, loc_i - 1)
    loc_i_max = min(i_max, loc_i + 2)
    loc_j_min = max(0, loc_j - 1)
    loc_j_max = min(j_max, loc_j + 2)

    local_grid = loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]
    local_grid = local_grid[local_grid != null_value].reshape(-1)
    center_value = loc_grid[loc_i, loc_j].astype(int)
    if local_grid.shape[0] == 0:
      majority = center_value
    else:
      count_list = np.bincount(local_grid.astype(int))
      if count_list.shape[0] == 2:
        other_value = int(1-center_value)
        if count_list[center_value] / (count_list[other_value] + count_list[center_value]) < 4/9:
          majority = other_value
        else:
          majority = center_value
      else:
        majority = np.bincount(local_grid.astype(int)).argmax()
      # majority = stats.mode(local_grid, keepdims=True)[0][0]

    loc_grid_new[loc_i, loc_j] = majority

  return loc_grid_new


def swap_partition(loc_grid, locs, pid):
  '''Used as a sub-function to improve contiguity.
  locs is an arrary of locations in partition pid (pid is either 0 or 1)
  '''
  i_max, j_max = loc_grid.shape
  loc_grid_new = np.copy(loc_grid)
  for loc in locs:
    loc_i = loc[0]
    loc_j = loc[1]
    count = 0
    count_eq = 0 #there are -1 values

    ##for 8-neighbor
    loc_i_min = max(0, loc_i - 1)
    loc_i_max = min(i_max, loc_i + 2)
    loc_j_min = max(0, loc_j - 1)
    loc_j_max = min(j_max, loc_j + 2)

    loc_mask = (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max] >= 0).astype(int)
    count = int(np.sum( (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]!=pid).astype(int) * loc_mask ))
    count_eq = int(np.sum( (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]==pid).astype(int) ))

    if count > count_eq:
      loc_grid_new[loc_i, loc_j] = np.abs(1-pid)

  return loc_grid_new


def grid_to_partitions(loc_grid, null_value = -1):
  group_id_grid = np.arange(loc_grid.shape[0] * loc_grid.shape[1])
  group_id_grid = group_id_grid.reshape(loc_grid.shape)

  s0 = group_id_grid[loc_grid==0]
  s1 = group_id_grid[loc_grid==1]

  return s0, s1


#customized spatial contiguity check, optional
def get_refined_partitions(s0, s1, y_val_gid, g_loc, dir = None, branch_id = None):
  '''This will refine partitions based on specific user needs, i.e., this function is problem-specific.
     This example will improve the spatial contiguity of the obtained partitions, by swapping partition assignments for an element that is surrounded by elements belonging to another partition.
     s0 and s1 contains group_ids
     g_loc contains locations of groups (can be represented by row id and column id)
  '''

  #s0, s1 do not contain gid directly
  s0_group = get_s_list_group_ids(s0.astype(int), y_val_gid).astype(int)
  s1_group = get_s_list_group_ids(s1.astype(int), y_val_gid).astype(int)

  #need to first got grid-based locations (for efficiency)

  #s0,s1 to grid
  loc_grid = -np.ones([np.max(g_loc[:,0])+1, np.max(g_loc[:,1])+1])#need to default values to -1
  loc_s0 = g_loc[s0_group]#need to test
  loc_s1 = g_loc[s1_group]
  loc_grid[loc_s0[:,0], loc_s0[:,1]] = 0
  loc_grid[loc_s1[:,0], loc_s1[:,1]] = 1

  # print(loc_grid)

  if dir is not None and branch_id is not None:
    generate_vis_image_for_all_groups(loc_grid, dir = dir, ext = '_debug1_' + branch_id, vmin = -1, vmax = 1)

  #only perform once after each partitioning, should not affect efficiency
  # loc_grid = swap_partition(loc_grid, loc_s0, 0)
  # loc_grid = swap_partition(loc_grid, loc_s1, 1)
  loc_grid = swap_partition_general(loc_grid, np.vstack([loc_s0, loc_s1]))

  if dir is not None:
    generate_vis_image_for_all_groups(loc_grid, dir = dir, ext = '_debug2_' + branch_id, vmin = -1, vmax = 1)
  # print(loc_grid)

  #grid to s0, s1
  s0_group, s1_group = grid_to_partitions(loc_grid)
  gid_to_s_map = get_gid_to_s_map(y_val_gid, s0, s1)
  s0, s1 = s_group_to_s(s0_group, s1_group, gid_to_s_map)

  return s0.astype(int), s1.astype(int)

def s_group_to_s(s0_group, s1_group, gid_to_s_map):
  s0 = gid_to_s_map[s0_group]
  s1 = gid_to_s_map[s1_group]
  return s0, s1

def get_top_cells(g, flex = FLEX_OPTION, flex_ratio = FLEX_RATIO, flex_type = FLEX_TYPE, cnt = None):
  '''get the top half cells with largest values (return values are row ids)'''

  #this is only to get index to be used for s0, s1
  sorted_g = np.argsort(g,0)#second input might not be needed
  sorted_g = sorted_g[::-1]
  # sorted_g = g.shape[0] - 1 - sorted_g#sorted_g[::-1]
  set_size = np.ceil(sorted_g.shape[0]/2).astype(int)

  if flex:
    if cnt is not None:
      # print('#Debug: np.stack([g, cnt]): ', np.stack([g, cnt]))
      cnt = cnt[sorted_g]
    set_size = optimize_size(g, set_size, flex_ratio, flex_type = flex_type, cnt = cnt)

  s0 = sorted_g[0:set_size]
  s1 = sorted_g[set_size:]



  return s0, s1

def get_score(y_true, y_pred, mode = MODE):

  score = None
  if mode == 'classification':
    if len(y_true.shape)==1:
      y_true = tf.one_hot(y_true, NUM_CLASS)
      y_pred = tf.one_hot(y_pred, NUM_CLASS)
    else:
    #this is to make coding consistent for later parts of the function (where tf functions are used)
      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)
      # tf.convert_to_tensor(numpy_array, dtype=tf.float32)

    #reshape image or time-series labels
    if len(y_true.shape)>=3:
      y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
      y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

    # y_pred = to_categorical(np.argmax(arr, axis=1), 3)
    score = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    score = score.numpy()

    #select_class
    #here need to remove the rows from non-selected classes
    #which will otherwise affect the diff margin sizes in sig_test()
    if SELECT_CLASS is not None:
      score_select = np.zeros(score.shape)
      for i in range(SELECT_CLASS.shape[0]):
        class_id = int(SELECT_CLASS[i])
        score_select[y_true.numpy()[:,class_id]==1] = 1
      score = score[score_select.astype(bool)]
      '''Check what if score is empty?
      '''

  else:
    score = tf.keras.losses.MSE(y_true, y_pred, reduction=tf.keras.losses.Reduction.NONE)#reduction=tf.keras.losses.Reduction.NONE
    score = - score.numpy()

  return score

def get_split_score(y0_true, y0_pred, y1_true, y1_pred, mode = MODE):
  score0 = get_score(y0_true, y0_pred)
  score1 = get_score(y1_true, y1_pred)

  score = np.hstack([score0, score1])

  return score0, score1

'''Partitioning optimization.'''
def scan(y_true_value, true_pred_value, min_sample,
         flex = FLEX_OPTION,
         flex_type = FLEX_TYPE,
         return_score = False):#connected = True, g_grid = None, X_dim = None, step_size = None,

  '''flex_type: determines if the split balance is based on #groups or #samples in groups'''
  c,b = get_c_b(y_true_value, true_pred_value)

  max_iteration = 1000

  #init q
  q = np.zeros(y_true_value.shape[1])
  q_init = np.nan_to_num(c/b)
  for i in range(q_init.shape[1]):
    q_class = q_init[:,i]
    # s_class, _ = get_top_cells(q_class)
    print('scan init step (ignore immediate "cnt = None" warning)')
    s_class, _ = get_top_cells(q_class)#no cnt-based flex for initialization?
    q[i] = np.sum(c[s_class,i]) / np.sum(b[s_class,i])

  # q = np.random.rand(y_true_value.shape[1])*2
  # q = np.exp(q)

  q_filter = np.sum(b,0) < min_sample
  q[q_filter == 1] = 1
  q[q == 0] = 1
  q = np.expand_dims(q,0)

  #prepare cnt for sample count based flex
  b_cnt = np.sum(b,1)

  log_lr_prev = 0
  for i in range(max_iteration):#coordinate descent
    #update location
    g = c * np.log(q) + b * (1-q)
    g = np.sum(g, 1)
    s0, s1 = get_top_cells(g, flex_type = flex_type, cnt = b_cnt)
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

    # print('s0', s0)
    # print('s1', s1)

  if return_score:
    return s0, s1, g
    # return s0, s1, g[s0], g[s1]
  else:
    return s0, s1


def get_refined_partitions_all(X_branch_id, s_branch, X_group, dir = None):
  '''This is used to refine partitions with all partition ids (not for smoothing binary partitions during the training process).'''

  unique_branch = np.unique(X_branch_id[X_branch_id != ''])
  branch_id_len = np.array(list(map(lambda x: len(x), unique_branch)))
  unique_branch = unique_branch[np.argsort(branch_id_len).astype(int)]
  #here grid has null/empty value of 0 (in partition_opt null is -1)
  grid, id_map = vis_partition_group(s_branch, unique_branch, step_size=STEP_SIZE, max_depth = MAX_DEPTH, return_id_map = True)
  id_map[0] = ''#unique branch no longer contains ''
  print('id_map', id_map)
  print('grid min: ', np.min(grid))
  print('grid.shape', grid.shape)

  generate_vis_image_from_grid(grid, dir, file_name = 'all_refined_before')

  locs = generate_groups_loc(X_DIM, STEP_SIZE)
  for refine_i in range(REFINE_TIMES):
    grid = swap_partition_general(grid, locs, null_value = 0)

  if dir is not None:
    generate_vis_image_from_grid(grid, dir, file_name = 'all_refined')

  list_branch_id_int = grid.reshape(-1).astype(int)
  # list_group_id = np.arange(grid.shape[0] * grid.shape[1])
  list_branch_id = np.asarray([id_map[int_id] for int_id in list_branch_id_int])
  # np.stack([list_group_id, list_branch_id], axis = -1)
  X_branch_id = list_branch_id[X_group.astype(int)]

  return X_branch_id
