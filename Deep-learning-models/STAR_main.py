# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np

'''STAR'''
'''Can be easily customized with the template'''
from model import DNNmodel#model is easily customizable
# from customize import generate_groups#can customize group definition
from data import load_data
from initialization import init_X_info
from helper import create_dir
from transformation import partition
# from visualization import *
'''All global parameters'''
from paras import *

'''Create directories'''
model_dir, dir, dir_ckpt = create_dir()
CKPT_FOLDER_PATH = dir_ckpt

'''Load data'''
X, y = load_data()
# load_sig_test_lookup_table()#Load csv containing the look-up table for critical values

'''Initialize location-related and training information. Can be customized.
      X_id stores data points' ids in the original X, and is used as a reference.
      X_set stores train-val-test assignments: train=0, val=1, test=2
      X_branch_id stores branch_ids (or, partion ids) of each data points. All init to route branch ''. Dynamically updated during training.
      X_group stores group assignment: customizable. In this example, groups are defined by grid cells in space.
'''
X_group, X_set, X_id, X_branch_id = init_X_info(X, y)


'''Train to stablize before starting the first data partitioning'''
train_list_init = np.where(X_set == 0)
model = DNNmodel(ckpt_path = dir_ckpt)
model.model_compile()
model.train(X[train_list_init], y[train_list_init], branch_id = '')#'' is the root branch (before any splits)
model.save('')#save root branch


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
                     max_depth = MAX_DEPTH)#partition data to subsets following homogeneous distributions

'''Save s_branch'''
# print(s_branch)
s_branch.to_pickle(dir + '/' + 's_branch.pkl')

'''Testing'''
test_list = np.where(X_set == 2)

# if partitioning is already available through previous training
# file_dir = dir + '/'
# s_branch = pd.read_pickle(file_dir + 's_branch.pkl')

pre, rec, f1 = model.predict_test(X[test_list], y[test_list], X_group[test_list], s_branch)
print('pre:', pre, 'rec:', rec, 'f1:', f1)
