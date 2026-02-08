# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
# @License: MIT License

import numpy as np
from config import STEP_SIZE#this is the grid cell size (not step size for image patch generation for semantic segmentation)

'''This function can be customized for different types of spatial or non-spatial data.
It only needs data to be pre-processed into groups.'''

class GroupGenerator():
  '''
  Generate groups (minimum spatial units) for partitioning in STAR.
  This generator is an example for grid-based group definitions,
  where a grid is overlaid on the study area and each grid cell defines one group.
  '''
  def __init__(self, xmin, xmax, ymin, ymax, step_size):
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.step_size = step_size

  def get_groups(self, X_loc):
    X_loc = X_loc.copy()
    X_loc[:,0] = X_loc[:,0] - self.xmin
    X_loc[:,1] = X_loc[:,1] - self.ymin

    X_loc_grid = np.floor(X_loc/self.step_size)
    n_rows = np.max(X_loc_grid[:,0])+1
    n_cols = np.max(X_loc_grid[:,1])+1
    X_group = X_loc_grid[:,0]*n_cols + X_loc_grid[:,1]

    return X_group

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

def generate_groups_nonimg_input(X_loc, step_size):
  '''from RF test: might be the same as generate_groups_from_raw_loc()'''
  X_loc_grid = np.floor(X_loc/step_size)
  n_rows = np.max(X_loc_grid[:,0])+1
  n_cols = np.max(X_loc_grid[:,1])+1
  # print(n_rows, n_cols)
  X_group = X_loc_grid[:,0]*n_cols + X_loc_grid[:,1]
  return X_group

def get_locs_of_groups(X_group, X_loc):
  n_group = np.max(X_group).astype(int) + 1
  group_loc = np.zeros((n_group.astype(int), 2))
  for i in range(X_group.shape[0]):
    group_id = X_group[i]
    #for debugging
    if group_loc[group_id,0] > 0 and group_loc[group_id, 0] != X_loc[i, 0]:
      print('#Bug: same group with diff locs: ', i, group_id, group_loc[group_id, :], X_loc[i, :])
    group_loc[group_id, :] = X_loc[i, :]
    group_loc = group_loc.astype(int)

  return group_loc

def generate_groups_loc(X_DIM, step_size):#X_DIM, X_loc,  = STEP_SIZE
  '''Used to store row and columns ids of groups for spatial contiguity refinement if needed.
     Corresponds to groups from generate_groups_from_raw_loc
  '''
  n_cols = int(np.floor(X_DIM[1]/step_size)+1)
  n_rows = int(np.floor(X_DIM[0]/step_size)+1)
  #X_loc_grid = np.floor(X_loc/step_size)
  #n_rows = int(np.max(X_loc_grid[:,0])+1)
  #n_cols = int(np.max(X_loc_grid[:,1])+1)
  group_loc = -np.ones([n_cols * n_rows, 2])
  group_loc[:, 0] = np.floor(np.arange(n_cols * n_rows) / n_cols)
  group_loc[:, 1] = np.arange(n_cols * n_rows) % n_cols

  return group_loc.astype(int)

def generate_group_id_for_test(X_test):
  '''Use if data points' group ids in test are not included in generate_groups() function.
    For example, test's group ids can be inferred by maximizing similarity to groups from training data.
  '''

  return X_group


''' customized groups assignment using county assignment
'''
def generate_groups_counties(X_loc):
  import os
  import requests
  import geopandas as gpd

  county_file = 'CountyShp.zip'

  if not os.path.exists(county_file):
    census_url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip'
    r = requests.get(census_url) # create HTTP response object
    with open('CountyShp.zip','wb') as f:
      f.write(r.content)

  county_gdf = gpd.read_file(county_file)
  county_gdf['id'] = county_gdf.index
  county_gdf = county_gdf[['geometry']]

  '''X_loc for RF have this offset needed to add to the X_loc to retrieve the actual lat and lon'''
  offset = np.array([24.54815386, -124.72499])
  X_loc_original = X_loc + offset

  X_loc_geom = gpd.points_from_xy(X_loc_original[:,1], X_loc_original[:,0])
  X_loc_original_gdf = gpd.GeoDataFrame(X_loc_original, geometry=X_loc_geom)
  X_loc_original_gdf.crs = 'EPSG:4269'

  X_loc_join = X_loc_original_gdf.sjoin(county_gdf, how='left')
  # X_loc_join[X_loc_join['index_right'].isna()].plot()
  '''points on the US boundaries are not assigned to any couty in the previous command. We use spatial join nearest to refine the join. '''
  X_loc_join[X_loc_join['index_right'].isna()] = X_loc_join[X_loc_join['index_right'].isna()].drop(columns='index_right').sjoin_nearest(county_gdf, how='left')
  X_loc_join['group'] = X_loc_join['index_right'].astype(int)
  X_loc_join = X_loc_join.drop(columns=['index_right'])

  X_group = X_loc_join['group'].values
  print('Assigned to County Map')
  # np.save('X_group_debug.npy', X_group)
  return X_group
