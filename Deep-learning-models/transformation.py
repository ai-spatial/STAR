# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from config import *
from helper import *
from train_branch import *
from partition_opt import *
from sig_test import *

import models

def partition(model, X, y,
                     X_group , X_set, X_id, X_branch_id,
                     max_depth = MAX_DEPTH,
                     **paras):

  '''
  **paras is for model-specific parameters (could be different for deep learning and traditional ML)
  maybe create two different versions?
  '''
  #branch_table is also something that can be returned if needed
  branch_table = np.zeros([2**max_depth, max_depth])
  branch_table[:,0:MIN_DEPTH] = 1 #-1

  #init s list: a dictionary
  #all all cells to initial branch with id==''
  #dict #rows must be consistent
  s_branch, max_size_needed = init_s_branch(n_groups = N_GROUPS)

  #update
  branch_size_valid = True

  for i in range(max_depth-1):

    #update
    if not branch_size_valid:
      print("Branch size too small! Returning...")
      continue

    num_branches = 2**i
    init_epoch_number = PRETRAIN_EPOCH + EPOCH_TRAIN * i
    print("Level %d --------------------------------------------------" % (i))

    #removed in the group-based version (originally used to gradually increase spatial resolution)
    # #update step_size for creating grid
    # step_size = STEP_SIZE / (2**np.floor(i/2))

    for j in range(num_branches):

      if branch_table[j,i] == 0:
        continue

      #get branch_id
      branch_id = get_branch_id_for(i,j)

      #print only
      print("Level %d -- branch: %s --------------------------------------------------" % (len(branch_id), branch_id) )

      #get branch data
      train_list = get_id_list(X_branch_id, X_set, branch_id, set_id = 0)
      val_list = get_id_list(X_branch_id, X_set, branch_id, set_id = 1)

      if val_list[0].shape[0]==0:
        print('error! branch_id: ' + branch_id)
        print('train_list size: %d' % (train_list[0].shape[0]))

      '''
      Partitioning starts.
      key return: X0_train, y0_train, X0_val, y0_val, X1_train, y1_train, X1_val, y1_val
      '''
      #get train val data for branch
      X_val = X[val_list]
      y_val = y[val_list]

      #change model: evaluation function
      y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)
      del X_val

      #get s list
      #outdated: gid stores row and column ids for grid cells (here y_val_gid and true_pred_gid are the same set, order should be the same but double check if needed)
      #gid stores group ids for grid cells (here y_val_gid and true_pred_gid are the same set, order should be the same but double check if needed)
      #value stores cell-wise sum of per-class stats
      #need to make results returned by groupby have the same order
      #change stat: get stats function

      if MODE == 'classification':
        (y_val_gid, y_val_value,
         true_pred_gid, true_pred_value) = get_class_wise_stat(y_val, y_pred_before,
                                                               X_group[val_list])
        del y_val
        #s0 and s1 returned by scan are at grid cell level
        s0, s1 = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE)
      elif MODE == 'regression':
        (y_val_gid, stat_value, 
        count_id, count_value) = get_class_wise_stat(y_val, y_pred_before,
                                                               X_group[val_list])
        del y_val
        s0, s1 = scan_regression(stat_value/count_value, MIN_SCAN_CLASS_SAMPLE)

      # print('y_val_gid: ', y_val_gid)

      s0_group = get_s_list_group_ids(s0, y_val_gid)
      s1_group = get_s_list_group_ids(s1, y_val_gid)
      (X0_train, y0_train, X0_val, y0_val,
       X1_train, y1_train, X1_val, y1_val,
       s0_train, s1_train, s0_val, s1_val) = get_branch_data_by_group(X, y, X_group,
                                                                  train_list, val_list, s0_group, s1_group)

      # if check_split_validity(X0_train, MIN_BRANCH_SAMPLE_SIZE) == 0:
      #   print('sample size too small! returning')
      #   continue
      # if check_split_validity(X1_train, MIN_BRANCH_SAMPLE_SIZE) == 0:
      #   print('sample size too small! returning')
      #   continue

      if check_split_validity(y0_train, MIN_BRANCH_SAMPLE_SIZE) == 0:
        branch_size_valid = False
        print('sample size too small! returning')
        continue
      if check_split_validity(y1_train, MIN_BRANCH_SAMPLE_SIZE) == 0:
        branch_size_valid = False
        print('sample size too small! returning')
        continue

       
      #train and eval hypothetical branches (i.e., have optimized partitions but need to perform sig test)
      print("Training new branches...")
      y0_pred, y1_pred = train_and_eval_two_branch(model, X0_train, y0_train, X0_val,
                                                  X1_train, y1_train, X1_val, branch_id)

      sig = 1
      #test only if a new split will give a depth > MIN_DEPTH
      #otherwise directly split
      if (len(branch_id)+1) > MIN_DEPTH:
        print("Training base branch:")

        #get base branch data
        X_train = auto_hv_stack(X0_train, X1_train)
        X_val = auto_hv_stack(X0_val, X1_val)
        y_train = auto_hv_stack(y0_train, y1_train)
        y_val = auto_hv_stack(y0_val, y1_val)

        #y_pred_before is used for sig test
        y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)

        #train and eval base branch
        y_pred = train_and_eval_using_merged_branch_data(model, X_train, y_train, X_val, branch_id)

        #align with y_preds' shape, otherwise (y_true - y_pred) will broadcast into 2D arrays (n x n)

        if len(y_val.shape) == 1 and MODE == 'regression':
          #update shapes for regression! The current version assumes y has shape (N,1) where N is the number of data points
          y_val = np.expand_dims(y_val, axis = 1)
          y0_val = np.expand_dims(y0_val, axis = 1)
          y1_val = np.expand_dims(y1_val, axis = 1)
          # y_pred = np.expand_dims(y_pred, axis = 1)

        #get stats
        base_score = get_score(y_val, y_pred)
        base_score_before = get_score(y_val, y_pred_before)
        split_score0, split_score1 = get_split_score(y0_val, y0_pred, y1_val, y1_pred)

        #sig test
        sig = sig_test(base_score, split_score0, split_score1, base_score_before)

      else:
        print("Smaller than MIN_DEPTH, split directly...")

      #decision
      if sig == 1:
        print("+ Split %s into %s, %s" % (branch_id, branch_id+'0', branch_id+'1') )

        '''
        update X_branch_id
        scan version uses pre-stored s0 and s1 lists for train and val
        '''

        #update branch_id
        X0_train_id, X1_train_id, X0_val_id, X1_val_id = get_branch_X_id(X_id, train_list, val_list, s0_train, s1_train, s0_val, s1_val)
        X0_id = np.hstack([X0_train_id, X0_val_id])
        X1_id = np.hstack([X1_train_id, X1_val_id])

        # print('X1_id.shape: ', X1_id.shape)

        if X0_id.shape[0]==0 or X1_id.shape[0]==0:
          print('error in getting branch_id! X0_id size: %d, X1_id size: %d' % (X0_id.shape[0], X1_id.shape[0]))

        X_branch_id = update_branch_id(X0_id, X_branch_id, branch_id + '0')
        X_branch_id = update_branch_id(X1_id, X_branch_id, branch_id + '1')

        # print('X_branch_id 1 count: ', np.where(X_branch_id == '1'))

        '''
        update s table
        need to make sure ids returned by groupby are consistent
        '''
        #np.empty is not empty!
        s0_grid_set = np.zeros(max_size_needed, dtype = np.int32)#, dtype = 'O'
        s1_grid_set = np.zeros(max_size_needed, dtype = np.int32)
        s0_grid_set[:s0.shape[0]] = y_val_gid[s0]
        s1_grid_set[:s1.shape[0]] = y_val_gid[s1]
        s_branch[branch_id + '0'] = s0_grid_set
        s_branch[branch_id + '1'] = s1_grid_set

        #update branch_table and score_table
        if i+1 < MAX_DEPTH:
          next_level_row_ids_for_new_branches = [branch_id_to_loop_id(branch_id+'0'), branch_id_to_loop_id(branch_id+'1')]
          branch_table[next_level_row_ids_for_new_branches, i+1] = 1

        # vis_partition_training(grid, branch_id)

      else:
        print("= Branch %s not split" % (branch_id) )

  return X_branch_id, branch_table, s_branch
