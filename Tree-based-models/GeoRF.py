# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-06-20
# @License: MIT License

import numpy as np
from scipy import stats

#GeoRF
#Can be customized with the template

# from models import DNNmodel, LSTMmodel, UNetmodel#model is easily customizable
from model_RF import RFmodel, save_single, predict_test_group_wise#model is easily customizable
from sklearn.ensemble import RandomForestClassifier
# from customize import generate_groups_nonimg_input#can customize group definition
from customize import *
from data import *
from initialization import init_X_info, init_X_info_raw_loc, init_X_branch_id, train_val_split
from helper import create_dir, open_dir, get_X_branch_id_by_group, get_filter_thrd
from transformation import partition
from visualization import *
from metrics import get_class_wise_accuracy, get_prf
from partition_opt import get_refined_partitions_all
#All global parameters
from config import *

import pandas as pd
import os
import argparse
import sys
import time
import logging

#search for "!!!" for places to potentially update
class GeoRF():
	def __init__(self,
							 #Geo-RF specific paras
							 min_model_depth = MIN_DEPTH,
							 max_model_depth = MAX_DEPTH,#max number of levels in bi-partitioning hierarchy (e.g., max=1 means can partition at most once)
							 dir = "",
							 #RF specific paras
							 n_trees_unit = 100, num_class = NUM_CLASS, max_depth=None,#this is max tree depth in RF
               random_state=5,
               n_jobs = N_JOBS,
               mode=MODE, name = 'RF', type = 'static',
               sample_weights_by_class = None
							 #unused paras
							 # path,!!!generate this using above dir
							 # max_model_depth = MAX_DEPTH,#moved to above
							 #increase_thrd = 0.05,
							 # max_new_forests,
							):
		#Geo-RF specifics: inputs (not including basic RF paras)
		self.min_model_depth = min_model_depth
		self.max_model_depth = max_model_depth#max partitioning depth
		self.model_dir = dir#currently unused
		#not from user inputs

		#Geo-RF specifics: outputs
		self.model = None
		#this is used to store models from GeoRF
		#[None] * (2**self.max_model_depth)#[]#len(list)
		# self.X_branch_id = None#can be derived, no need to store
		self.branch_table = None
		self.s_branch = None
		#RF inputs
		self.n_trees_unit = n_trees_unit#number of trees for each model piece, see self.model
		self.num_class = num_class
		self.max_depth = max_depth
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.mode = mode#!!!may not be needed
		self.name = name
		self.type = type#static
		self.sample_weights_by_class = sample_weights_by_class
		#self.max_new_forests = max_new_forests#unused
		# self.path = path#!!!this may not be needed (can be generated using geo-RF path)

		#Create directories to store models and results
		if MODEL_CHOICE == 'RF':
			folder_name_ext = 'GeoRF'
		else:
			folder_name_ext = 'DL'#deep learning version

		separate_vis = True
		if separate_vis:
			#model_dir: main folder for the experiment
		  #dir: folder to store geo-rf related intermediate results such as space partitions
		  #dir_ckpt: store trained models for different local models
		  #dir_vis: for visualzation
			model_dir, dir_space, dir_ckpt, dir_vis = create_dir(folder_name_ext = folder_name_ext, separate_vis = separate_vis)
		else:
			model_dir, dir_space, dir_ckpt = create_dir(folder_name_ext = folder_name_ext)
			dir_vis = dir_space

		CKPT_FOLDER_PATH = dir_ckpt#might not be used

		self.model_dir = model_dir
		self.dir_space = dir_space
		self.dir_ckpt = dir_ckpt
		self.dir_vis = dir_vis

		#toggle between prints
		self.original_stdout = sys.stdout


	'''Needs for new functions!!!
		done* Define a function to generate X_set using VAL_RATIO
		done (generate_groups_nonimg_input(X_loc, step_size) in customize)* Define a function to generate X_group for grid settings
		done* Define a function to only generate X_id.
		done (leave in config)* Add attributes such as: flex_ratio (if passed as paras)
		* Parameters such as min & max_depth can only show up either in config file or in georf definition,
			otherwise inconsistencies may occur. If they are only used as default, that's okay but need to
			make sure the values are passed to the functions when called.
		* Remove image generations in training
	'''

	#Train GeoRF
	def fit(self, X, y, X_group, X_set = None, val_ratio = VAL_RATIO, print_to_file = True):#X_loc is unused
		"""
    Train the geo-aware random forest (Geo-RF).

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Output targets.
		X_group: array-like
				Provides a group ID for each data point in X. The groups are groupings of locations,
				which serve two important purposes:
				(1) Minimum spatial unit: A group is the minimum spatial unit for space-partitioning
				(or just data partitioning if non-spatial data). For example, a grid/fishnet can be used
				to generate groups,	where all data points in each grid cell belong to one group. As a
				minimum spatial unit,	all points in the same group will always be placed in the same
				spatial partition.
				(2) Test point model selection: Once Geo-RF is trained, the groups are used to determine
				which local model a test point should use. First, the group ID of a test point is determined
				by its location (e.g., based on grid cells), and then the corresponding partition ID of the
				group is used to determine the local RF to use for the prediction (all groups in a spatial
				partition share the same local model.).
		X_set : array-like
        Optional. One value per data point in X, with 0 for training and 1 for validation. If this
				is not provided, val_ratio will be used to randomly assign points to validation set. In Geo-RF,
				this left-out validation set is used by default to evaluate the necessity of new partitions. It
				can be used as a usual validation set, or if desired, a separate validation set can be used for
				other hyperparameter tuning, that are independent to this set, which is not used as training samples
				but their evaluations are used in the GeoRF branching process.
		val_ratio: float
				Optional. Used if X_set is not provided to assign samples to the validation set.
    Returns
    -------
		georf: GeoRF class object
				Returns GeoRF model parameters, trained results (e.g., partitions) and pointers to trained weights.
				The trained results are needed to make spatially-explicit predictions.
    """

		#Logging: testing purpose only
		logging.basicConfig(filename=self.model_dir + '/' + "model.log",
						format='%(asctime)s %(message)s',
						filemode='w')
		logger=logging.getLogger()
		logger.setLevel(logging.INFO)

		#print to file
		if print_to_file:
			print('model_dir:', self.model_dir)
			print_file = self.model_dir + '/' + 'log_print.txt'
			sys.stdout = open(print_file, "w")

		print('Options: ')
		print('CONTIGUITY & REFINE_TIMES: ', CONTIGUITY, REFINE_TIMES)
		print('MIN_BRANCH_SAMPLE_SIZE: ', MIN_BRANCH_SAMPLE_SIZE)
		print('FLEX_RATIO: ', FLEX_RATIO)
		print('Partition MIN_DEPTH & MAX_DEPTH: ', MIN_DEPTH, MAX_DEPTH)

		print('X.shape: ', X.shape)
		print('y.shape: ', y.shape)


		# #for debugging#X_loc removed here
		# print(np.min(X_loc[:,0]), np.max(X_loc[:,0]))
		# print(np.min(X_loc[:,1]), np.max(X_loc[:,1]))

		#Initialize location-related and training information. Can be customized.
    #X_id stores data points' ids in the original X, and is used as a reference.
    #X_set stores train-val-test assignments: train=0, val=1, test=2
    #X_branch_id stores branch_ids (or, partion ids) of each data points. All init to route branch ''. Dynamically updated during training.
    #X_group stores group assignment: customizable. In this example, groups are defined by grid cells in space.
		if X_set is None:
			X_set = train_val_split(X, val_ratio=val_ratio)
		X_id = np.arange(X.shape[0])#the id is used to later refer back to the original X, and the related information
		X_branch_id = init_X_branch_id(X, max_depth = self.max_model_depth)
		# X_group, X_set, X_id, X_branch_id = init_X_info_raw_loc(X, y, X_loc, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, step_size = STEP_SIZE, predefined = PREDEFINED_GROUPS)

		# '''RF paras''' --> unused
		# max_new_forests = [1,1,1,1,1,1]
		# sample_weights_by_class = None#np.array([0.05, 0.95])#None#np.array([0.05, 0.95])#None

		#timer
		start_time = time.time()

		#Train to stablize before starting the first data partitioning
		train_list_init = np.where(X_set == 0)
		if MODEL_CHOICE == 'RF':
			#RF
			self.model = RFmodel(self.dir_ckpt, self.n_trees_unit, max_depth = self.max_depth)#can add sample_weights_by_class
			self.model.train(X[train_list_init], y[train_list_init], branch_id = '', sample_weights_by_class = self.sample_weights_by_class)

		self.model.save('')#save root branch

		print("Time single: %f s" % (time.time() - start_time))
		logger.info("Time single: %f s" % (time.time() - start_time))

		#Spatial transformation (data partitioning, not necessarily for spatial data).
		#This will automatically partition data into subsets during training, so that each subset follows a homogeneous distribution.
		#format of branch_id: for example: '0010' refers to a branch after four bi-partitionings (four splits),
		  #and 0 or 1 shows the partition it belongs to after each split.
		  #'' is the root branch (before any split).
		#s_branch: another key output, that stores the group ids for all branches.
		#X_branch_id: contains the branch_id for each data point.
		#branch_table: shows which branches are further split and which are not.
		X_branch_id, self.branch_table, self.s_branch = partition(self.model, X, y,
		                   X_group , X_set, X_id, X_branch_id,
		                   min_depth = self.min_model_depth, max_depth = self.max_model_depth)#X_loc = X_loc is unused

		#Save s_branch
		print(self.s_branch)
		self.s_branch.to_pickle(self.dir_space + '/' + 's_branch.pkl')
		np.save(self.dir_space + '/' + 'X_branch_id.npy', X_branch_id)
		np.save(self.dir_space + '/' + 'branch_table.npy', self.branch_table)

		print("Time: %f s" % (time.time() - start_time))
		logger.info("Time: %f s" % (time.time() - start_time))

		#update branch_id for test data
		X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)#should be the same (previously fixed some potential inconsistency)

		#Optional: Improving Spatial Contiguity
		#The default function only works for groups defined by a grid, where majority voting in local neighborhoods are used to remove
		#fragmented partitions (e.g., one grid cell with a different partition ID from most of its neighbors).'''
		## GLOBAL_CONTIGUITY = False#unused (mentioned in visualization part later)
		if CONTIGUITY:
			X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = self.dir_vis, min_component_size = MIN_COMPONENT_SIZE)
		## 	GLOBAL_CONTIGUITY = True#unused

		if print_to_file:
			sys.stdout.close()
			sys.stdout = self.original_stdout

		return self

	def predict(self, X, X_group, save_full_predictions = False):
		"""
    Evaluating GeoRF and/or RF.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Output targets.
		X_group: array-like
				Same way of assignment as training. See detailed explanations in training.
		save_full_predictions: boolean
				Optional. If True, save predictions to file.
    Returns
    -------
		y_pred: array-like
				Returns predictions.
    """

		#Model assignment
		X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
		y_pred = self.model.predict_georf(X, X_group, self.s_branch, X_branch_id = X_branch_id)

		if save_full_predictions:
			np.save(self.dir_space + '/' + 'y_pred_georf.npy', y_pred)

		return y_pred

	def evaluate(self, Xtest, ytest, Xtest_group, eval_base = False, print_to_file = True):
		"""
    Evaluating GeoRF and/or RF.

    Parameters
    ----------
    Xtest: array-like
        Input features.
    ytest: array-like
        Output targets.
		Xtest_group: array-like
				Same way of assignment as training. See detailed explanations in training.
		eval_base: boolean
				Optional. If True, base RF will be evaluated for comparison.
		print_to_file: boolean
				Optional. If True, prints will go to file.
    Returns
    -------
		pre, rec, f1: array-like
				Precision, recall and F1 scores. Separate for different classes in arrays.
		pre_single, rec_single, f1_single: array-like
				If eval_base is True, additionally returns results for the base RF model.
    """

		#Logging: testing purpose only.
		logging.basicConfig(filename=self.model_dir + '/' + "model_eval.log",
						format='%(asctime)s %(message)s',
						filemode='w')
		logger=logging.getLogger()
		logger.setLevel(logging.INFO)

		#print to file
		if print_to_file:
			print('model_dir:', self.model_dir)
			print('Printing to file.')
			print_file = self.model_dir + '/' + 'log_print_eval.txt'
			sys.stdout = open(print_file, "w")

		#Geo-RF
		start_time = time.time()

		#Model assignment
		Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)

		pre, rec, f1, total_class = self.model.predict_test(Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)
		print('f1:', f1)
		log_print = ', '.join('%f' % value for value in f1)
		logger.info('f1: %s' % log_print)
		logger.info("Pred time: GeoRF: %f s" % (time.time() - start_time))

		#Base RF
		if eval_base:
			start_time = time.time()
			self.model.load('')
			y_pred_single = self.model.predict(Xtest)
			true_single, total_single, pred_total_single = get_class_wise_accuracy(ytest, y_pred_single, prf = True)
			pre_single, rec_single, f1_single, total_class = get_prf(true_single, total_single, pred_total_single)

			print('f1_base:', f1_single)
			log_print = ', '.join('%f' % value for value in f1_single)
			logger.info('f1_base: %s' % log_print)
			logger.info("Pred time: Base: %f s" % (time.time() - start_time))

			if print_to_file:
				sys.stdout.close()
				sys.stdout = self.original_stdout

			return pre, rec, f1, pre_single, rec_single, f1_single

		if print_to_file:
			sys.stdout.close()
			sys.stdout = self.original_stdout

		return pre, rec, f1

	def visualize_grid(self, Xtest, ytest, Xtest_group, step_size = STEP_SIZE):
		'''Visualization: temporary for testing purposes.
		Combine into a function later.

		Parameters
    ----------
    Xtest: array-like
        Input features.
    ytest: array-like
        Output targets.
		Xtest_group: array-like
				Same way of assignment as training. See detailed explanations in training.
		step_size: float
				Used to generate grid (must be same as the one used to generate grid-based groups).
    Returns
    -------
		None.
		'''

		#Model assignment
		Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)
		results, groups, total_number = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)

		#visualize partitions
		generate_vis_image(self.s_branch, Xtest_branch_id, max_depth = self.max_model_depth, dir = self.dir_vis, step_size = step_size)

		for class_id_input in SELECT_CLASS:
			class_id = int(class_id_input)
			ext = str(class_id)
			grid, vmin, vmax = generate_performance_grid(results, groups, class_id = class_id, step_size = step_size)
			print('X_DIM, grid.shape: ', X_DIM, grid.shape)
			grid_count, vmin_count, vmax_count = generate_count_grid(total_number, groups, class_id = class_id, step_size = step_size)
			generate_vis_image_for_all_groups(grid, dir = self.dir_vis, ext = '_star' + ext, vmin = vmin, vmax = vmax)
			generate_vis_image_for_all_groups(grid_count, dir = self.dir_vis, ext = '_count' + ext, vmin = vmin_count, vmax = vmax_count)

			results_base, groups_base, _ = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, base = True, X_branch_id = Xtest_branch_id)
			grid_base, vmin_base, vmax_base = generate_performance_grid(results_base, groups_base, class_id = class_id, step_size = step_size)
			generate_vis_image_for_all_groups(grid_base, dir = self.dir_vis, ext = '_base' + ext, vmin = vmin_base, vmax = vmax_base)

			cnt_vis_thrd = get_filter_thrd(grid_count, ratio = 0.2)
			grid_diff, vmin_diff, vmax_diff = generate_diff_grid((grid - grid_base)*(grid_count>=cnt_vis_thrd), groups, step_size = step_size)
			generate_vis_image_for_all_groups(grid_diff, dir = self.dir_vis, ext = '_diff' + ext, vmin = vmin_diff, vmax = vmax_diff)

			np.save(self.dir_space + '/' + 'grid' + ext + '.npy', grid)
			np.save(self.dir_space + '/' + 'grid_base' + ext + '.npy', grid_base)
			np.save(self.dir_space + '/' + 'grid_count' + ext + '.npy', grid_count)

		return
