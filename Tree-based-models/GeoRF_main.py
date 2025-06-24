# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-06-20
# @License: MIT License

'''
GeoRF_main.py

This file shows an example on how to define, train and evaluate a GeoRF model.
The general usage is similar to standard machine learning models, with fit(), predict() and evaluate() functions.
The main difference is that GeoRF is a spatially-explicit model, which means location info is needed as part of inputs.

Usage:
		There are 4 key steps:
			(1) Data loading: Prepare standard X, y, as well as locations X_loc
			(2) Define groups using GroupGenerator() or by customizing it for your own data: This is needed in GeoRF, which defines the minimum spatial unit for space partitioning.
					For example, a grid can be used to split the study area into groups, where all data points in each grid cell form a group.
					The data may contain thousands of groups, which may be partitioned by GeoRF into 10-20 partitions (one RF per partition). This is just to give a sense of the quantities.
					In testing, the grouping is used to assign test data points to local models (one local RF model is learned for each partition).
			(3) Training: georf.fit()
			(4) Prediction and evaluation: georf.predict() and georf.evaluate()

		For visualizations, code for grid-based grouping is provided. For other group definitions, please write customized visualization functions.
'''

import numpy as np
from scipy import stats
import pandas as pd
import os
import argparse
import sys

#GeoRF
#Can be customized with the template
from GeoRF import GeoRF
from customize import GroupGenerator
from data import load_demo_data#load_data_us_cdl
from helper import get_spatial_range
from initialization import train_test_split_all
#All global parameters
from config import *



if __name__ == '__main__':

	#load data (crop classification data): please customize for your own data
	#X_loc stores coordinates (e.g., lat and lon) for each data point in X
	# X, y, X_loc = load_data_us_cdl(full = True, from_raw = False, crop_type = 'corn', onehot = False)
	#Demo data is a sampled version (50%) and uses 13 out of 333 features to reduce the storage cost and execution time.
	X, y, X_loc = load_demo_data(crop_type='corn')

	#prepare groups (minimum spatial units for space-partitioning and location groupings, e.g., using a grid)
	xmin, xmax, ymin, ymax = get_spatial_range(X_loc)#the X_DIM in config assumes min is 0.
	group_gen = GroupGenerator(xmin, xmax, ymin, ymax, STEP_SIZE)
	X_group = group_gen.get_groups(X_loc)

	#train test split
	# Xtrain, ytrain, Xtest, ytest = train_test_split(X, y, test_ratio = TEST_RATIO)
	(Xtrain, ytrain, Xtrain_loc, Xtrain_group,
	Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_all(X, y, X_loc, X_group, test_ratio = TEST_RATIO)

	#define georf
	#max_model_depth: max depth of the hierarchical partitioning in GeoRF
	#max_depth: max depth of each tree in random forest (10 used in demo to reduce execution time)
	#n_jobs: number of cores to use to run the model in parallel
	georf = GeoRF(min_model_depth = MIN_DEPTH,	max_model_depth = MAX_DEPTH, n_jobs = N_JOBS, max_depth=10)
	#definition with all parameters
	#dir: defaulted to "result" in code (auto-generated), no need to change
	# georf = GeoRF(min_model_depth = MIN_DEPTH,	max_model_depth = MAX_DEPTH, dir = "",
	# 							n_trees_unit = 100, num_class = NUM_CLASS, max_depth=None, random_state=5, n_jobs = N_JOBS,
	# 							mode=MODE, name = 'RF', type = 'static', sample_weights_by_class = None)

	#training
	georf.fit(Xtrain, ytrain, Xtrain_group, val_ratio = VAL_RATIO)

	#preidct
	# ypred = georf.predict(Xtest, Xtest_group)

	#evaluatipn
	(pre, rec, f1,
	pre_base, rec_base, f1_base) = georf.evaluate(Xtest, ytest, Xtest_group, eval_base = True, print_to_file = True)

	#visualization (grid version)
	#!!! This is for testing purposes only with some hard-coded vars (only works for the case
	#when xmin and ymin in spatial range are both 0) and not yet developed to be used for other purposes
	#here locations are inferred from groups and X_DIM
	georf.visualize_grid(Xtest, ytest, Xtest_group, step_size = STEP_SIZE)
