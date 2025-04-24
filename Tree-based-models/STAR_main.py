# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
from scipy import stats

'''STAR'''
'''Can be easily customized with the template'''
# from models import DNNmodel, LSTMmodel, UNetmodel#model is easily customizable
from model_RF import RFmodel, save_single, predict_test_group_wise#model is easily customizable
from sklearn.ensemble import RandomForestClassifier
# from customize import generate_groups_nonimg_input#can customize group definition
from customize import *
from data import *
from initialization import init_X_info, init_X_info_raw_loc
from helper import create_dir, open_dir, get_X_branch_id_by_group
from transformation import partition
from visualization import *

from metrics import get_class_wise_accuracy, get_prf

from partition_opt import get_refined_partitions_all

# from data_syn_exp import *

import pandas as pd
# import geopandas as gpd
import os


'''All global parameters'''
from paras import *

import argparse

import sys



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--crop', default='corn', choices=['soybean', 'corn', 'wheat', 'cotton', 'rice', 'multi'])
	parser.add_argument('--n_jobs', default=N_JOBS, type=int)
	parser.add_argument('--n_trees', default=100, type=int)
	parser.add_argument('--max_tree_depth', default=None, type = int)
	# parser.add_argument('--max_tree_depth', default=10, type = int)
	parser.add_argument('--min_partition_depth', default=MIN_DEPTH, type=int)
	parser.add_argument('--max_partition_depth', default=MAX_DEPTH, type=int)
	parser.add_argument('--train_ratio', default=TRAIN_RATIO, type=float)
	parser.add_argument('--val_ratio', default=VAL_RATIO, type=float)
	parser.add_argument('--step_size', default=STEP_SIZE, type=float)
	parser.add_argument('--min_component_size', default=10, type=float)
	args = parser.parse_args()

	crop_type = args.crop
	n_jobs = args.n_jobs
	N_TREES = args.n_trees
	MAX_TREE_DEPTH = args.max_tree_depth
	MIN_DEPTH = args.min_partition_depth
	MAX_DEPTH = args.max_partition_depth
	TRAIN_RATIO = args.train_ratio
	VAL_RATIO = args.val_ratio
	STEP_SIZE = args.step_size
	MIN_COMPONENT_SIZE = args.min_component_size

	print_to_file = True

	'''meta-comment: combine all common reshape operations to one function.
	'''

	'''Create directories'''
	if MODEL_CHOICE == 'RF':
		#for RF or ensemble
		folder_name_ext = crop_type + '_' + str(N_TREES) + '_' + EVAL_EXT
	else:
		#for DL
		folder_name_ext = crop_type + '_' + 'DL' + '_' + EVAL_EXT

	# model_dir, dir, dir_ckpt = create_dir(folder_name_ext = folder_name_ext)
	separate_vis = True
	if separate_vis:
		model_dir, dir, dir_ckpt, dir_vis = create_dir(folder_name_ext = folder_name_ext, separate_vis = separate_vis)
	else:
		model_dir, dir, dir_ckpt = create_dir(folder_name_ext = folder_name_ext)
		dir_vis = dir

	CKPT_FOLDER_PATH = dir_ckpt#might not be used

	'''Open existing directories'''
	# model_dir = 'result_auto_10'
	# dir, dir_ckpt = open_dir(model_dir)

	'''Load data'''
	# X, y = load_data()
	X, y, X_loc = load_data_us_cdl(full = True, from_raw = False, crop_type = crop_type, onehot = ONEHOT)

	if TIME_SERIES:#the data contains spectral info for about a year (can use single month or entire sequence)
		all_time_feature = N_TIME_FEATURE * N_TIME
		X_other = X[:,all_time_feature:(all_time_feature + N_OTHER_FEATURE)]
		X = X[:,0:all_time_feature]
		X = X.reshape([-1, N_TIME, N_TIME_FEATURE])
		X_other = np.expand_dims(X_other, 1)
		X_other = np.tile(X_other, [1,N_TIME,1])
		X = np.concatenate([X, X_other], axis = -1)
		print('X.shape: (time)', X.shape)
		# X = X.reshape([-1, (N_TIME_FEATURE + N_OTHER_FEATURE)])
		# print('X.shape: (flat)', X.shape)

	'''Logging: testing purpose only.'''
	import logging
	logging.basicConfig(filename=model_dir + '/' + "model.log",
					format='%(asctime)s %(message)s',
					filemode='w')
	logger=logging.getLogger()
	logger.setLevel(logging.INFO)

	#print to file
	if print_to_file:
		print(model_dir)
		print_file = model_dir + '/' + 'log_print.txt'
		sys.stdout = open(print_file, "w")
		print(model_dir)

	print('Options: ')
	print('CONTIGUITY & REFINE_TIMES: ', CONTIGUITY, REFINE_TIMES)
	print('MIN_BRANCH_SAMPLE_SIZE: ', MIN_BRANCH_SAMPLE_SIZE)
	print('FLEX_RATIO: ', FLEX_RATIO)
	print('Partition MIN_DEPTH & MAX_DEPTH: ', MIN_DEPTH, MAX_DEPTH)

	print('X.shape: ', X.shape)
	print('y.shape: ', y.shape)


	#for debugging
	print(np.min(X_loc[:,0]), np.max(X_loc[:,0]))
	print(np.min(X_loc[:,1]), np.max(X_loc[:,1]))
	# load_sig_test_lookup_table()#Load csv containing the look-up table for critical values


	'''Initialize location-related and training information. Can be customized.
	    X_id stores data points' ids in the original X, and is used as a reference.
	    X_set stores train-val-test assignments: train=0, val=1, test=2
	    X_branch_id stores branch_ids (or, partion ids) of each data points. All init to route branch ''. Dynamically updated during training.
	    X_group stores group assignment: customizable. In this example, groups are defined by grid cells in space.
	'''
	# X_group, X_set, X_id, X_branch_id = init_X_info(X, y)
	# X_group, X_set, X_id, X_branch_id = init_X_info_raw_loc(X, y, X_loc, train_ratio = TRAIN_RATIO, step_size = STEP_SIZE)
	X_group, X_set, X_id, X_branch_id = init_X_info_raw_loc(X, y, X_loc, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, step_size = STEP_SIZE, predefined = PREDEFINED_GROUPS)
	# group_loc = get_locs_of_groups(X_group, X_loc)

	'''RF paras'''
	max_new_forests = [1,1,1,1,1,1]
	sample_weights_by_class = None#np.array([0.05, 0.95])#None#np.array([0.05, 0.95])#None

	#timer
	import time
	start_time = time.time()

	'''Train to stablize before starting the first data partitioning'''
	train_list_init = np.where(X_set == 0)
	if MODEL_CHOICE == 'RF':
		#RF
		model = RFmodel(dir_ckpt, N_TREES, max_new_forests, max_depth = MAX_TREE_DEPTH)#can add sample_weights_by_class
		model.train(X[train_list_init], y[train_list_init], branch_id = '', sample_weights_by_class = sample_weights_by_class)#removed in def: sample_weights_by_class = sample_weights_by_class
	# else:
	# 	if MODEL_CHOICE == 'DNN':
	# 		model = DNNmodel(path = dir_ckpt)
	# 	if MODEL_CHOICE == 'LSTM':
	# 		model = LSTMmodel(path = dir_ckpt)
	# 	model.model_compile()
	# 	model.train(X[train_list_init], y[train_list_init], branch_id = '')#'' is the root branch (before any splits)

	model.save('')#save root branch

	print("Time single: %f s" % (time.time() - start_time))
	logger.info("Time single: %f s" % (time.time() - start_time))

	'''Spatial transformation (data partitioning, not necessarily for spatial data).
	This will automatically partition data into subsets during training, so that each subset follows a homogeneous distribution.
	format of branch_id: for example: '0010' refers to a branch after four bi-partitionings (four splits),
	  and 0 or 1 shows the partition it belongs to after each split.
	  '' is the root branch (before any split).
	s_branch: another key output, that stores the group ids for all branches.
	X_branch_id: contains the branch_id for each data point.
	branch_table: shows which branches are further split and which are not.
	'''
	X_branch_id, branch_table, s_branch = partition(model, X, y,
	                   X_group , X_set, X_id, X_branch_id,
										 #group_loc = group_loc,
                                         X_loc = X_loc,
	                   min_depth = MIN_DEPTH, max_depth = MAX_DEPTH)#partition data to subsets following homogeneous distributions

	'''Save s_branch'''
	print(s_branch)
	s_branch.to_pickle(dir + '/' + 's_branch.pkl')
	np.save(dir + '/' + 'X_branch_id.npy', X_branch_id)
	np.save(dir + '/' + 'branch_table.npy', branch_table)

	print("Time: %f s" % (time.time() - start_time))
	logger.info("Time: %f s" % (time.time() - start_time))

	#update branch_id for test data
	X_branch_id = get_X_branch_id_by_group(X_group, s_branch)#should be the same (previously fixed some potential inconsistency)

	'''Optional: Improving Spatial Contiguity'''
	GLOBAL_CONTIGUITY = False
	if CONTIGUITY:
		X_branch_id = get_refined_partitions_all(X_branch_id, s_branch, X_group, dir = dir_vis, min_component_size = MIN_COMPONENT_SIZE)
		GLOBAL_CONTIGUITY = True



	'''Testing'''

	'''If using an existing model without training'''
	# model = UNetmodel(ckpt_path = dir_ckpt)#careful with name overload (model is used as a package, though with from ... import ...)
	#
	# import pandas as pd
	# s_branch = pd.read_pickle(dir + '/' + 's_branch.pkl')

	# from models import predict_test, predict_test_one_branch
	test_list = np.where(X_set == 2)

	start_time = time.time()

	print_full_predictions = True

	'''STAR'''
	pre, rec, f1, total_class = model.predict_test(X[test_list], y[test_list], X_group[test_list], s_branch, X_branch_id = X_branch_id[test_list])
	# pre, rec, f1, total_class = model.predict_test(X[test_list], y[test_list], X_group[test_list], s_branch)
	print('f1:', f1)
	log_print = ', '.join('%f' % value for value in f1)
	logger.info('f1: %s' % log_print)

	logger.info("Pred time: STAR: %f s" % (time.time() - start_time))
	start_time = time.time()

	if print_full_predictions:
		y_pred_georf = model.predict_georf(X[test_list], X_group[test_list], s_branch, X_branch_id = X_branch_id[test_list])
		y_pred_georf_full = model.predict_georf(X, X_group, s_branch, X_branch_id = X_branch_id)

	'''Base'''
	model.load('')
	y_pred_single = model.predict(X[test_list])
	true_single, total_single, pred_total_single = get_class_wise_accuracy(y[test_list], y_pred_single, prf = True)
	pre_single, rec_single, f1_single, total_class = get_prf(true_single, total_single, pred_total_single)

	if print_full_predictions:
		y_pred_single_full = model.predict(X)
		np.save(dir + '/' + 'y_pred_georf.npy', y_pred_georf)
		np.save(dir + '/' + 'y_pred_georf_full.npy', y_pred_georf_full)
		np.save(dir + '/' + 'y_pred_single.npy', y_pred_single)
		np.save(dir + '/' + 'y_pred_single_full.npy', y_pred_single_full)

	print('f1_single:', f1_single)
	log_print = ', '.join('%f' % value for value in f1_single)
	logger.info('f1_single: %s' % log_print)

	logger.info("Pred time: Base: %f s" % (time.time() - start_time))
	# start_time = time.time()


	'''Visualization: temporary for testing purposes.
	# Combine into a function later.'''
	if not PREDEFINED_GROUPS:
		if not GLOBAL_CONTIGUITY:
			generate_vis_image(s_branch, X_branch_id, max_depth = MAX_DEPTH, dir = dir_vis, step_size = STEP_SIZE)
			# generate_vis_image_from_grid(grid, dir, file_name = 'all_refined')


		results, groups, total_number = predict_test_group_wise(model, X[test_list], y[test_list], X_group[test_list], s_branch, X_branch_id = X_branch_id[test_list])

		for class_id_input in SELECT_CLASS:
			class_id = int(class_id_input)
			ext = str(class_id)
			grid, vmin, vmax = generate_performance_grid(results, groups, class_id = class_id, step_size = STEP_SIZE)
			print('X_DIM, grid.shape: ', X_DIM, grid.shape)
			grid_count, vmin_count, vmax_count = generate_count_grid(total_number, groups, class_id = class_id, step_size = STEP_SIZE)
			generate_vis_image_for_all_groups(grid, dir = dir_vis, ext = '_star' + ext, vmin = vmin, vmax = vmax)
			generate_vis_image_for_all_groups(grid_count, dir = dir_vis, ext = '_count' + ext, vmin = vmin_count, vmax = vmax_count)

			results_base, groups_base, _ = predict_test_group_wise(model, X[test_list], y[test_list], X_group[test_list], s_branch, base = True, X_branch_id = X_branch_id[test_list])
			grid_base, vmin_base, vmax_base = generate_performance_grid(results_base, groups_base, class_id = class_id, step_size = STEP_SIZE)
			generate_vis_image_for_all_groups(grid_base, dir = dir_vis, ext = '_base' + ext, vmin = vmin_base, vmax = vmax_base)

			grid_diff, vmin_diff, vmax_diff = generate_diff_grid((grid - grid_base)*(grid_count>=100), groups, step_size = STEP_SIZE)
			generate_vis_image_for_all_groups(grid_diff, dir = dir_vis, ext = '_diff' + ext, vmin = vmin_diff, vmax = vmax_diff)

			np.save(dir + '/' + 'grid' + ext + '.npy', grid)
			np.save(dir + '/' + 'grid_base' + ext + '.npy', grid_base)
			np.save(dir + '/' + 'grid_count' + ext + '.npy', grid_count)



	else:
		generate_vis_image_shp(X_branch_id, X_loc, dir_vis, step_size = STEP_SIZE)

	if print_to_file:
		sys.stdout.close()
