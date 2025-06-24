# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
# import tensorflow as tf
import pandas as pd

from config import *

#visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from PIL import Image

# import geopandas as gpd

'''
Functions in visualization normally need to be customized depending on how the groups are generated, and the data type.
'''

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


def vis_partition(s_branch, unique_branch, max_depth = MAX_DEPTH, step_size = STEP_SIZE):

  #get cell size
  step_size = step_size / (2**np.floor((MAX_DEPTH-1)/2))
  grid, n_row, n_col = generate_grid_vis(X_DIM, step_size)

  color_id = 0
  for branch_id in unique_branch:#list(s_branch.keys()):
    current_depth = len(branch_id)
    if current_depth >= max_depth:
      continue

    if current_depth == 0:
      current_step_size = step_size
    else:
      current_step_size = step_size / (2**np.floor((current_depth-1)/2))

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


def vis_partition_group(s_branch, unique_branch, step_size, max_depth = MAX_DEPTH, return_id_map = False):
  '''id map stores the 1 to 1 matching between branch_id (e.g., '01') and color_id (an integer id for each branch)'''

  #get cell size
  #!!!be careful, this may need to be updated with np.floor + 1 to be consistent with other initializations
  #n_rows = np.ceil(X_DIM[0]/step_size).astype(int)
  #n_cols = np.ceil(X_DIM[1]/step_size).astype(int)
  n_rows = np.floor(X_DIM[0]/step_size).astype(int) + 1
  n_cols = np.floor(X_DIM[1]/step_size).astype(int) + 1
  grid = np.zeros((n_rows, n_cols))

  id_map = {}

  color_id = 1#background color to be 0
  for branch_id in unique_branch:#list(s_branch.keys()):
    current_depth = len(branch_id)
    if current_depth >= max_depth or current_depth == 0:#0 means '', X_branch_id may contain '' as test samples are have not been assigned branches yet
      continue

    # if current_depth == 0:
    #   current_step_size = STEP_SIZE
    # else:
    #   current_step_size = STEP_SIZE / (2**np.floor((current_depth-1)/2))

    # step_size_ratio = current_step_size / step_size
    step_size_ratio = 1

    # if current_depth == 2 or current_depth == 4:
    #   step_size_ratio = step_size_ratio * 2

    # print('s_branch[branch_id]', s_branch[branch_id])

    for gid in s_branch[branch_id]:

      if gid is None or (gid == -1):#(gid == 0):#check this, a valid gid may be 0!!!
        #break#probably some are removed by contiguity (swap small components)
        continue

      # print(gid)
      row_id = np.floor(gid/n_cols).astype(int)
      col_id = (gid % n_cols).astype(int)
      # row_id = gid[0]
      # col_id = gid[1]

      #get row ranges in full resolution grid
      row_min = row_id * step_size_ratio
      row_max = row_min + step_size_ratio
      col_min = col_id * step_size_ratio
      col_max = col_min + step_size_ratio

      row_range = np.arange(row_min,row_max).astype(int)
      col_range = np.arange(col_min,col_max).astype(int)

      grid[np.ix_(row_range, col_range)] = color_id

    id_map[int(color_id)] = branch_id
    color_id = color_id + 1

  if return_id_map:
    return grid, id_map
  else:
    return grid

def generate_vis_image(s_branch, X_branch_id, max_depth, dir, step_size = STEP_SIZE, file_name='all'):
  print(list(s_branch.keys()))

  unique_branch = np.unique(X_branch_id)
  branch_id_len = np.array(list(map(lambda x: len(x), unique_branch)))
  unique_branch = unique_branch[np.argsort(branch_id_len).astype(int)]
  print(unique_branch)

  from PIL import ImageOps
  VIS_SIZE = 1000
  grid = vis_partition_group(s_branch, unique_branch, step_size=step_size, max_depth = max_depth)
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img, interpolation="none")#cmap=color_palette,
  cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  #the following line has issues with .py in some environments (might be version issues)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  # fig.savefig(dir + '/' + 'all.png')
  fig.savefig(dir + '/' + file_name + '.png')
  np.save(dir + '/' + 'grid' + file_name + '.npy', grid)


def generate_vis_image_from_grid(grid, dir, file_name='all'):
  from PIL import ImageOps
  VIS_SIZE = 1000
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  color_palette = sns.color_palette('deep', len(np.unique(grid.reshape(-1))))
  im = plt.imshow(resized_img, interpolation="none")#cmap=color_palette,
  cbar = plt.colorbar(im, ticks = np.arange(len(np.unique(grid.reshape(-1)))))
  #the following line has issues with .py in some environments (might be version issues)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  # fig.savefig(dir + '/' + 'all.png')
  fig.savefig(dir + '/' + file_name + '.png')


def generate_performance_grid(results, groups, step_size = STEP_SIZE, prf = True, class_id = None, X_dim = X_DIM):
  '''
  Args:
    results: group-wise performances (e.g., prf from the predict_test_group_wise())
    groups: group ids for the results
    prf: True if "results" is for multi-outputs (e.g., prf results with pre, rec, f1, total number); and False if for other scalar outputs, e.g., accuracy, etc.
    class_id: if only wants to show results for one class (e.g., background and one-class)
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  grid = np.empty((n_rows, n_cols))
  grid[:,:] = np.nan

  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  if prf:
    if class_id is None:
      class_weights = results[:,3,:] / np.expand_dims(np.sum(results[:,3,:], axis = -1), 1)
      grid[row_ids, col_ids] = np.sum(results[:,2,:] * class_weights, axis = -1)
    else:
      grid[row_ids, col_ids] = results[:, 2, class_id]
  else:
    grid[row_ids, col_ids] = results

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  return grid, vmin, vmax

def generate_count_grid(results, groups, step_size = STEP_SIZE, class_id = None, X_dim = X_DIM):
  '''
  Args:
    results: group-wise performances (e.g., prf from the predict_test_group_wise())
    groups: group ids for the results
    prf: True if "results" is for multi-outputs (e.g., prf results with pre, rec, f1, total number); and False if for other scalar outputs, e.g., accuracy, etc.
    class_id: if only wants to show results for one class (e.g., background and one-class)
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  grid = np.empty((n_rows, n_cols))
  grid[:,:] = np.nan

  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  if class_id is None:
    grid[row_ids, col_ids] = np.sum(results, axis = -1)
  else:
    grid[row_ids, col_ids] = results[:, class_id]

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  print('count: results.shape: ', results.shape)
  print('count: grid.shape: ', grid.shape)
  print('count: results: ', results)
  print('count: np.sum(results, axis = -1): ', np.sum(results, axis = -1))

  return grid, vmin, vmax


def get_symmetric_vmin_vmax(vmin, vmax, option = 'always'):
  if option == 'always':
    v = max(abs(vmin), abs(vmax))
    vmin = -v
    vmax = v
  elif vmax>0 and vmin<0:
      v = max(abs(vmin), abs(vmax))
      vmin = -v
      vmax = v

  return vmin, vmax

def generate_diff_grid(grid, groups, step_size = STEP_SIZE, X_dim = X_DIM):
  '''
  Adhoc function used to get vmin and vmax for visualization. The grid is ready from performance grids.
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  vmin, vmax = get_symmetric_vmin_vmax(vmin, vmax)

  return grid, vmin, vmax



def generate_vis_image_for_all_groups(grid, dir, ext = '', vmin = None, vmax = None):
  '''This generates visualization images for all groups.
  Args:
    grid: from generate_grid_vis()
    ext: to append to the default file name
    vmin: min value for clim in matplotlib
    vmax: max value for clim in matplotlib
  '''

  # if vmin is None:
  #   vmin = np.min(grid)
  # if vmax is None:
  #   vmax = np.max(grid)

  from PIL import ImageOps
  VIS_SIZE = 1000
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  resized_img = np.ma.array(resized_img, mask=np.isnan(resized_img))
  cmap = matplotlib.cm.viridis
  cmap.set_bad('white',1.)#mask out nan values

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  # color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img, interpolation="none", cmap = cmap)#cmap=color_palette,
  if vmin is not None:
    # vmin, vmax = get_symmetric_vmin_vmax(vmin, vmax)
    im.set_clim(vmin,vmax)
  cbar = plt.colorbar(im)
  # cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  #the following line has issues with .py in some environments (might be version issues)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  fig.savefig(dir + '/' + 'result_group' + ext + '.png')
