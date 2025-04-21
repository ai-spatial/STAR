# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd
from scipy import stats

from paras import *
from helper import *
from train_branch import *
from partition_opt import *
from sig_test import *

from customize import *

from visualization import *

import models

def partition(model, X, y,
                     X_group , X_set, X_id, X_branch_id,
                     #group_loc = None,
                     X_loc = None,#this is optional for spatial smoothing (partition shape refinement)
                     min_depth = MIN_DEPTH,
                     max_depth = MAX_DEPTH,
                     refine_times = REFINE_TIMES,
                     **paras):

  '''
  **paras is for model-specific parameters (could be different for deep learning and traditional ML)
  maybe create two different versions?
  '''
  #branch_table is also something that can be returned if needed
  branch_table = np.zeros([2**max_depth, max_depth])
  branch_table[:,0:min_depth] = 1 #-1

  #init s list: a dictionary
  #all all cells to initial branch with id==''
  #dict #rows must be consistent
  s_branch, max_size_needed = init_s_branch(n_groups = N_GROUPS)#grid_dim = GRID_DIM

  for i in range(max_depth-1):

    num_branches = 2**i
    init_epoch_number = PRETRAIN_EPOCH + EPOCH_TRAIN * i
    print("Level %d --------------------------------------------------" % (i))

    #removed in the group-based version
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
      partition
      key return: X0_train, y0_train, X0_val, y0_val, X1_train, y1_train, X1_val, y1_val
      '''
      #get train val data for branch
      X_val = X[val_list]
      y_val = y[val_list]

      #change model: evaluation function
      y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)
      del X_val

      #get s list
      #gid stores row and column ids for grid cells (here y_val_gid and true_pred_gid are the same set, order should be the same but double check if needed)
      #value stores cell-wise sum of per-class stats
      #need to make results returned by groupby have the same order
      #change stat: get stats function

      (y_val_gid, y_val_value,
       true_pred_gid, true_pred_value) = get_class_wise_stat(y_val, y_pred_before,
                                                             X_group[val_list])
                                                            #  X_loc[np.ix_(val_list[0], GRID_COLS)])#X_val_grid

      '''Verify if there are still data of interest left for selected class.
      y_val_value should have shape (num_groups, n_class)'''
      # if np.sum(y_val_value) <= MIN_BRANCH_SAMPLE_SIZE:
      if (np.sum(y_val_value) <= MIN_BRANCH_SAMPLE_SIZE): #and i>=min_depth:
        print('selected classes: sample size too small! returning')
        continue
      # print('y_val_gid: ', y_val_gid)

      del y_val
      #s0 and s1 returned by scan are at grid cell level
      #correct
      # s0, s1 = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE)
      #incorrect

      RETURN_SCAN_SCORE = True
      if RETURN_SCAN_SCORE:
        # the error is calculated in get_c_b, no need to convert true to error here
        s0, s1, gscore = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE, return_score = RETURN_SCAN_SCORE)
        # s0, s1, gscore = scan(y_val_value, y_val_value * (1-true_pred_value), MIN_SCAN_CLASS_SAMPLE, return_score = RETURN_SCAN_SCORE)
      else:
        s0, s1 = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE)
        # s0, s1 = scan(y_val_value, y_val_value * (1-true_pred_value), MIN_SCAN_CLASS_SAMPLE)

      # s0_prev = s0
      # s1_prev = s1

      if CONTIGUITY:
        group_loc = generate_groups_loc(X_DIM, STEP_SIZE)
        #group_loc = generate_groups_loc(X_loc, STEP_SIZE)
        # refine_times = REFINE_TIMES
        for i_refine in range(refine_times):
          #!!!s0 and s1 do not contain gid; instead, they contain indices from y_val_true (for gid need to use y_val_gid)
          s0, s1 = get_refined_partitions(s0, s1, y_val_gid, group_loc, dir = model.path, branch_id = branch_id)

      #debug
      # group_loc = generate_groups_loc(X_DIM, STEP_SIZE)
      # #group_loc = generate_groups_loc(X_loc, STEP_SIZE)
      # #s0_group = get_s_list_group_ids(s0, y_val_gid)
      # #s1_group = get_s_list_group_ids(s1, y_val_gid)
      # s0_debug, s1_debug = get_refined_partitions(s0, s1, y_val_gid, group_loc, dir = model.path, branch_id = branch_id)
      # print('#Debug: s0, s1; s0_refine, s1_refine: ', s0.shape, s1.shape, s0_debug.shape, s1_debug.shape)

      s0_group = get_s_list_group_ids(s0, y_val_gid)
      s1_group = get_s_list_group_ids(s1, y_val_gid)
      (X0_train, y0_train, X0_val, y0_val,
       X1_train, y1_train, X1_val, y1_val,
       s0_train, s1_train, s0_val, s1_val) = get_branch_data_by_group(X, y, X_group,
                                                                  train_list, val_list, s0_group, s1_group)
      # print('s0: ', s0)
      # print('s1: ', s1)
      # print('s0_group: ', s0_group)
      # print('s1_group: ', s1_group)

      # if (check_split_validity(X0_train, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X1_train, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X0_val, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X1_val, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue

      if (check_split_validity(X0_train, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X1_train, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X0_val, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X1_val, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue

      #train and eval hypothetical branches
      print("Training new branches...")#i and j splits are not used in train_and_eval_two_branch()
      y0_pred, y1_pred = train_and_eval_two_branch(model, X0_train, y0_train, X0_val,
                                                  X1_train, y1_train, X1_val, branch_id)

      sig = 1
      #test only if a new split will give a depth > MIN_DEPTH
      #otherwise directly split
      if (len(branch_id)+1) > min_depth or model.name == 'RF':
        print("Training base branch:")

        #get base branch data
        X_train = auto_hv_stack(X0_train, X1_train)
        X_val = auto_hv_stack(X0_val, X1_val)
        y_train = auto_hv_stack(y0_train, y1_train)
        y_val = auto_hv_stack(y0_val, y1_val)

        #y_pred_before is used for sig test
        y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)

        #align with y_preds' shape, otherwise (y_true - y_pred) will broadcast into 2D arrays (n x n)
        if len(y_val.shape) == 1 and MODE == 'regression':
          #update shapes for regression! The current version assumes y has shape (N,1) where N is the number of data points
          #addition: the expand_dims is also necessary for Reduction.None to function correctly in MSE calcualtion
          y_val = np.expand_dims(y_val, axis = 1)
          y0_val = np.expand_dims(y0_val, axis = 1)
          y1_val = np.expand_dims(y1_val, axis = 1)
          # y_pred = np.expand_dims(y_pred, axis = 1)

        #get stats
        base_score_before = get_score(y_val, y_pred_before)
        split_score0, split_score1 = get_split_score(y0_val, y0_pred, y1_val, y1_pred)

        #additional step for random forest
        #evaluate if previous fuller branch may outperform one sub-branch
        if model.name == 'RF':
          #for quick testing of design
          #not considering efficiency here
          if MODE == 'classification':
            y0_pred_before = base_eval_using_merged_branch_data(model, X0_val, branch_id)
            y1_pred_before = base_eval_using_merged_branch_data(model, X1_val, branch_id)
            split_score0_before, split_score1_before = get_split_score(y0_val, y0_pred_before, y1_val, y1_pred_before)
            print('effects of using only data in partition:')
            print('score before 0: ', np.mean(split_score0_before), 'score after 0: ', np.mean(split_score0))
            print('score before 1: ', np.mean(split_score1_before), 'score after 1: ', np.mean(split_score1))
            if np.mean(split_score0_before) >= np.mean(split_score0) and np.mean(split_score1_before) < np.mean(split_score1):
              #overwrite
              split_score0 = split_score0_before
              model.load(branch_id)
              model.save(branch_id + '0')
              print('overwrite branch', branch_id + '0', ' weights with branch', branch_id)

            elif np.mean(split_score0_before) < np.mean(split_score0) and np.mean(split_score1_before) >= np.mean(split_score1):
              #overwrite
              split_score1 = split_score1_before
              model.load(branch_id)
              model.save(branch_id + '1')
              print('overwrite branch', branch_id + '1', ' weights with branch', branch_id)
            elif ((len(branch_id)+1) <= min_depth and
                  np.mean(split_score0_before) >= np.mean(split_score0) and
                  np.mean(split_score1_before) >= np.mean(split_score1)):
              sig = 0

        #train and eval base branch
        if model.type == 'incremental':
          y_pred = train_and_eval_using_merged_branch_data(model, X_train, y_train, X_val, branch_id)
          base_score = get_score(y_val, y_pred)
        else:
          y_pred = None
          base_score = base_score_before#np.zeros(base_score_before.shape)

        #sig test
        if (len(branch_id)+1) > min_depth:
          #RF model will get into evaluation regardless of min_depth for weight selection
          sig = sig_test(base_score, split_score0, split_score1, base_score_before)
        else:
          print("Smaller than MIN_DEPTH, split directly...")

      else:
        #only DL model may reach here
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

        print('X1_id.shape: ', X1_id.shape)

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
        # s0_grid_set = np.zeros(max_size_needed, dtype = np.int32)#, dtype = 'O'
        # s1_grid_set = np.zeros(max_size_needed, dtype = np.int32)
        s0_grid_set = -np.ones(max_size_needed, dtype = np.int32)#, dtype = 'O'
        s1_grid_set = -np.ones(max_size_needed, dtype = np.int32)
        s0_grid_set[:s0.shape[0]] = y_val_gid[s0]
        s1_grid_set[:s1.shape[0]] = y_val_gid[s1]
        s_branch[branch_id + '0'] = s0_grid_set#y_val_gid is at grid-cell level, and covers all cells in this branch
        s_branch[branch_id + '1'] = s1_grid_set

        #update branch_table and score_table
        if i+1 < max_depth:
          next_level_row_ids_for_new_branches = [branch_id_to_loop_id(branch_id+'0'), branch_id_to_loop_id(branch_id+'1')]
          branch_table[next_level_row_ids_for_new_branches, i+1] = 1

        # vis_partition_training(grid, branch_id)
        generate_vis_image(s_branch, X_branch_id, max_depth = max_depth, dir = model.path, step_size = STEP_SIZE, file_name = branch_id + '_split')
        #accuracy
        grid, vmin, vmax = generate_count_grid(true_pred_value/(y_val_value+0.0001), y_val_gid, class_id = 0, step_size = STEP_SIZE)
        generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_acc_' + branch_id, vmin = vmin, vmax = vmax)
        #gscore (used in scan statistics)
        # print('#Debug: true_pred_value.shape, y_val_gid.shape: ', true_pred_value.shape, y_val_gid.shape)
        scan_gscore = np.expand_dims(np.hstack([gscore[s0], gscore[s1]]), axis = -1)
        grid, vmin, vmax = generate_count_grid(scan_gscore, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_scan_' + branch_id, vmin = vmin, vmax = vmax)
        #gscore rank: ranking of each group in scan
        gscore_argsort = np.argsort(gscore,0)[::-1]
        gscore_rank = np.arange(gscore.shape[0])
        gscore_rank = gscore_rank[gscore_argsort.argsort()]
        #gscore_rank = gscore.shape[0] - 1 - gscore_rank
        scan_gscore_rank = np.expand_dims(np.hstack([gscore_rank[s0], gscore_rank[s1]]), axis = -1)
        grid, vmin, vmax = generate_count_grid(scan_gscore_rank, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_rank_' + branch_id, vmin = vmin, vmax = vmax)
        #gscore>0
        scan_gscore_positive = (scan_gscore>0).astype(int)
        grid, vmin, vmax = generate_count_grid(scan_gscore_positive, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_binary_' + branch_id, vmin = vmin, vmax = vmax)
        #count
        grid, vmin, vmax = generate_count_grid((y_val_value > MIN_GROUP_POS_SAMPLE_SIZE_FLEX).astype(int), y_val_gid, class_id = 0, step_size = STEP_SIZE)
        generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_cnt_' + branch_id, vmin = vmin, vmax = vmax)

      else:
        print("= Branch %s not split" % (branch_id) )

  return X_branch_id, branch_table, s_branch
