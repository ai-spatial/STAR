# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-09
# @License: MIT License
#
# PyTorch version of GeoDL_main.py — same workflow, uses PyTorch models via GeoDL_torch.
#
# Usage: same as GeoDL_main.py. Ensure PyTorch is installed: pip install torch

import numpy as np
from scipy import stats
import pandas as pd
import os
import argparse
import sys

from GeoDL_torch import GeoDL
from customize import GroupGenerator
from data import load_demo_data_seg
from helper import get_spatial_range
from initialization import train_test_split_all
from config import *

# Optional: load_demo_data if you have it
try:
    from data import load_demo_data
except ImportError:
    load_demo_data = None


if __name__ == '__main__':
    # Load data (customize for your data)
    if load_demo_data is not None:
        X, y, X_loc = load_demo_data()
    else:
        X, y, X_loc = load_demo_data_seg(return_loc=True)

    xmin, xmax, ymin, ymax = get_spatial_range(X_loc)
    group_gen = GroupGenerator(xmin, xmax, ymin, ymax, STEP_SIZE)
    X_group = group_gen.get_groups(X_loc)

    (Xtrain, ytrain, Xtrain_loc, Xtrain_group,
     Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_all(
        X, y, X_loc, X_group, test_ratio=TEST_RATIO)

    geodl = GeoDL(model_choice=MODEL_CHOICE)
    geodl.fit(Xtrain, ytrain, Xtrain_group, val_ratio=VAL_RATIO)

    if MODE == 'classification':
        pre, rec, f1 = geodl.evaluate(Xtest, ytest, Xtest_group)
        print('pre:', pre, 'rec:', rec, 'f1:', f1)
    elif MODE == 'regression':
        err_abs, err_square = geodl.evaluate(Xtest, ytest, Xtest_group)
        print('err_abs:', err_abs)
        print('err_square:', err_square)
