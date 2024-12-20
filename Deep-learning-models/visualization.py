# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from paras import *

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#in training visualization
def vis_partition_training(grid, branch_id):
  '''Visualize space-partitionings.'''

  vis_size = 256
  # grid_img = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
  resized_img = Image.fromarray(grid).resize((vis_size, vis_size), Image.NEAREST)
  resized_img_np = np.array(resized_img)

  fig = plt.figure(figsize=(3,3))
  # color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img_np, interpolation="none")#cmap=color_palette,
  # cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  # cbar.ax.set_yticklabels(list(s_branch.keys()))
  img_name = branch_id
  if len(branch_id) == 0:
    img_name = 'initial'

  fig.savefig(dir + '/' + img_name + '.png')

  return#grid

def generate_grid_vis(X_dim, step_size):#, max_size_decay

  n_row = np.ceil(X_dim[0]/step_size).astype(int)
  n_col = np.ceil(X_dim[1]/step_size).astype(int)

  grid = np.ones([n_row, n_col])

  return grid, n_row, n_col

def vis_partition(s_branch, unique_branch, max_depth = MAX_DEPTH):

  #get cell size
  step_size = STEP_SIZE / (2**np.floor((MAX_DEPTH-1)/2))
  grid, n_row, n_col = generate_grid_vis(X_DIM, step_size)

  color_id = 0
  for branch_id in unique_branch:#list(s_branch.keys()):
    current_depth = len(branch_id)
    if current_depth >= max_depth:
      continue

    if current_depth == 0:
      current_step_size = STEP_SIZE
    else:
      current_step_size = STEP_SIZE / (2**np.floor((current_depth-1)/2))

    step_size_ratio = current_step_size / step_size

    # if current_depth == 2 or current_depth == 4:
    #   step_size_ratio = step_size_ratio * 2

    for gid in s_branch[branch_id]:

      if gid is None or (gid == 0):
        break

      # print(gid)
      row_id = gid[0]
      col_id = gid[1]

      #get row ranges in full resolution grid
      row_min = row_id * step_size_ratio
      row_max = row_min + step_size_ratio
      col_min = col_id * step_size_ratio
      col_max = col_min + step_size_ratio

      row_range = np.arange(row_min,row_max).astype(int)
      col_range = np.arange(col_min,col_max).astype(int)

      grid[np.ix_(row_range, col_range)] = color_id

    color_id = color_id + 1

  return grid


# if __name__ == "__main__":
#     VIS_SIZE = 1000
#
#     print('s_branch shape: ' + str(s_branch.shape))
#     print(list(s_branch.keys()))
#
#     #for colorbar
#     # max_depth = 6
#     test_data_grid = test_data_partition(s_branch, max_depth = max_depth)
#     unique_branch = np.unique(test_data_grid)
#     branch_id_len = np.array(list(map(lambda x: len(x), unique_branch)))
#     unique_branch = unique_branch[np.argsort(branch_id_len).astype(int)]
#     print(unique_branch)
#
#     grid = vis_partition(s_branch, unique_branch, max_depth = max_depth)
#     resized_img = Image.fromarray(grid).resize((VIS_SIZE, VIS_SIZE), Image.NEAREST)
#     resized_img = np.array(resized_img)
#
#     # print(grid)
#     # print(resized_img)
#
#     fig = plt.figure(figsize=(6,6))
#     color_palette = sns.color_palette('deep', s_branch.shape[1])
#     im = plt.imshow(resized_img, interpolation="none")#cmap=color_palette,
#     cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
#     cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())
#
#     fig.savefig(dir + '/' + 'all.png')
