# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
from paras import STEP_SIZE#this is the grid cell size (not step size for image patch generation for semantic segmentation)

'''This function can be customized for different types of spatial or non-spatial data.
It only needs data to be pre-processed into groups.'''

def generate_groups(X_loc):
  '''Create groups of data points.
  This is needed for data partitioning optimization, as it needs to calculalte statistics at the group level.
  In this example, we use each grid cell as a group.
  Users can customize their own groups for spatial or non-spatail data by modifying this function.

  Notes for customization:
  1. Each group should contain sufficient number of data points for statistics calculation (e.g., >50).
  2. Data points in each group should follow roughly the same distribution (e.g., nearby data points in space).

  Args in this example:
    X_loc: Locations of data points. First two values are pixel locations.
        The other two are grid cell locations (all pixels in a grid cell shares the same grid cell location).
  '''

  n_rows = np.max(X_loc[:,2])+1
  n_cols = np.max(X_loc[:,3])+1
  X_group = X_loc[:,2]*n_cols + X_loc[:,3]

  return X_group

def generate_groups_from_raw_loc(X_loc, step_size = STEP_SIZE):
  '''Create groups of data points. This version only uses raw locations of each data point.
  '''
  # n_rows = np.max(X_loc[:,2])+1
  n_cols = np.floor(np.max(X_loc[:,1])/step_size)+1
  X_group = np.floor(X_loc[:,0]/step_size) * n_cols  + np.floor(X_loc[:,1]/step_size)

  return X_group
