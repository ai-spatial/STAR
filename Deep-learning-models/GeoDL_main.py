# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

'''
GeoDL_main.py

This file shows an example on how to define, train and evaluate a GeoDL model.
The general usage is similar to standard machine learning models, with fit(), predict() and evaluate() functions.
The main difference is that GeoDL is a spatially-explicit model, which means location info is needed as part of inputs.

Usage:
    There are 4 key steps:
      (1) Data loading: Prepare standard X, y, as well as locations X_loc
      (2) Define groups using GroupGenerator() or by customizing it for your own data: This is needed in GeoDL, which defines the minimum spatial unit for space partitioning.
          For example, a grid can be used to split the study area into groups, where all data points in each grid cell form a group.
          The data may contain thousands of groups, which may be partitioned by GeoDL into 10-20 partitions (one model per partition). This is just to give a sense of the quantities.
          In testing, the grouping is used to assign test data points to local models (one local model is learned for each partition).
      (3) Training: geodl.fit()
      (4) Prediction and evaluation: geodl.predict() and geodl.evaluate()
'''

import numpy as np
from scipy import stats
import pandas as pd
import os
import argparse
import sys

#GeoDL
#Can be customized with the template
from GeoDL import GeoDL
from customize import GroupGenerator
from data import load_demo_data_seg
from helper import get_spatial_range
from initialization import train_test_split_all
#All global parameters
from config import *


if __name__ == '__main__':

	#load data (crop classification data): please customize for your own data
	#X_loc stores coordinates (e.g., row and col) for each data point in X
	#For segmentation demo data:
	# X, y, X_loc = load_demo_data_seg(return_loc = True)
  X, y, X_loc = load_demo_data()

	#prepare groups (minimum spatial units for space-partitioning and location groupings, e.g., using a grid)
	xmin, xmax, ymin, ymax = get_spatial_range(X_loc)#the X_DIM in config assumes min is 0.
	group_gen = GroupGenerator(xmin, xmax, ymin, ymax, STEP_SIZE)
	X_group = group_gen.get_groups(X_loc)

	#train test split
	(Xtrain, ytrain, Xtrain_loc, Xtrain_group,
	Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_all(X, y, X_loc, X_group, test_ratio = TEST_RATIO)

	#define geodl
	geodl = GeoDL(model_choice = MODEL_CHOICE)

	#training
	geodl.fit(Xtrain, ytrain, Xtrain_group, val_ratio = VAL_RATIO)

	#predict
	# ypred = geodl.predict(Xtest, Xtest_group)

	#evaluation
	if MODE == 'classification':
		pre, rec, f1 = geodl.evaluate(Xtest, ytest, Xtest_group)
		print('pre:', pre, 'rec:', rec, 'f1:', f1)
	elif MODE == 'regression':
		err_abs, err_square = geodl.evaluate(Xtest, ytest, Xtest_group)
		print('err_abs:', err_abs)
		print('err_square:', err_square)
