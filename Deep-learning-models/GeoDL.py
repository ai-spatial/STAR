# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-06-20
# @License: MIT License

import numpy as np
import sys
import time
import logging

from config import *
from models import DNNmodel, UNetmodel#, predict_test, predict_test_group_wise
from initialization import init_X_branch_id, train_val_split
from helper import create_dir, open_dir, get_X_branch_id_by_group, get_filter_thrd
from transformation import partition
from visualization import generate_vis_image, generate_performance_grid, generate_count_grid, generate_vis_image_for_all_groups, generate_diff_grid
from partition_opt import get_refined_partitions_all
from metrics import get_class_wise_accuracy, get_prf


class GeoDL():
  """
  Geo-aware deep learning model with STAR partitioning.
  Provides a class-based API similar to GeoRF: fit, predict, evaluate.
  """
  def __init__(self,
               model_choice = MODEL_CHOICE,
               model_kwargs = None,
               model_dir = ""):
    self.model_choice = model_choice
    self.model_kwargs = model_kwargs or {}
    self.model_dir = model_dir
    self.max_depth = MAX_DEPTH

    if self.model_dir:
      self.dir_space, self.dir_ckpt = open_dir(self.model_dir)
    else:
      self.model_dir, self.dir_space, self.dir_ckpt = create_dir(model_dir = MODEL_DIR)

    self.model = self._build_model()
    self.model.model_compile()
    self.branch_table = None
    self.s_branch = None
    self.original_stdout = sys.stdout

  def _build_model(self):
    if self.model_choice.lower() in ['unet', 'u-net']:
      return UNetmodel(ckpt_path = self.dir_ckpt, **self.model_kwargs)
    return DNNmodel(ckpt_path = self.dir_ckpt, **self.model_kwargs)

  def fit(self, X, y, X_group, X_set = None, val_ratio = VAL_RATIO,
          max_depth = MAX_DEPTH, print_to_file = True):
    """
    Train STAR with deep learning models.
    """
    self.max_depth = max_depth
    if X_set is None:
      X_set = train_val_split(X, val_ratio = val_ratio)

    X_id = np.arange(X.shape[0])
    X_branch_id = init_X_branch_id(X, max_depth = max_depth)

    if print_to_file:
      print_file = self.model_dir + '/' + 'log_print.txt'
      sys.stdout = open(print_file, "w")
      print('model_dir:', self.model_dir)

    # Train to stabilize before starting the first data partitioning
    train_list_init = np.where(X_set == 0)
    self.model.train(X[train_list_init], y[train_list_init], branch_id = '')
    self.model.save('')

    # Spatial transformation (data partitioning)
    X_branch_id, self.branch_table, self.s_branch = partition(
      self.model, X, y, X_group, X_set, X_id, X_branch_id,
      max_depth = max_depth)

    # Optional: Improving Spatial Contiguity
    if CONTIGUITY:
      X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = self.dir_space, min_component_size = MIN_COMPONENT_SIZE, max_depth = max_depth)

    # Save results
    self.s_branch.to_pickle(self.dir_space + '/' + 's_branch.pkl')
    np.save(self.dir_space + '/' + 'X_branch_id.npy', X_branch_id)
    np.save(self.dir_space + '/' + 'branch_table.npy', self.branch_table)

    if print_to_file:
      sys.stdout.close()
      sys.stdout = self.original_stdout

    return self

  def predict(self, X, X_group, save_full_predictions = False):
    if self.s_branch is None:
      raise ValueError("Model not trained. Call fit() first.")
    y_pred = self.model.predict_geodl(X, X_group, self.s_branch)
    if save_full_predictions:
      np.save(self.dir_space + '/' + 'y_pred_geodl.npy', y_pred)
    return y_pred

  def evaluate(self, X, y, X_group, prf = True, eval_base = False, print_to_file = True):
    if self.s_branch is None:
      raise ValueError("Model not trained. Call fit() first.")
    logging.basicConfig(filename=self.model_dir + '/' + "model_eval.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)

    if print_to_file:
      print('model_dir:', self.model_dir)
      print('Printing to file.')
      print_file = self.model_dir + '/' + 'log_print_eval.txt'
      sys.stdout = open(print_file, "w")

    start_time = time.time()
    Xtest_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
    if MODE == 'classification':
      pre, rec, f1, total_class = self.model.predict_test(X, y, X_group, self.s_branch, X_branch_id = Xtest_branch_id)
      print('f1:', f1)
      log_print = ', '.join('%f' % value for value in f1)
      logger.info('f1: %s' % log_print)
      logger.info("Pred time: GeoDL: %f s" % (time.time() - start_time))

    elif MODE == 'regression':
      err_abs, err_square = self.model.predict_test(X, y, X_group, self.s_branch, X_branch_id = Xtest_branch_id)
      print('err_abs:', err_abs)
      print('err_square:', err_square)
      
      if np.isscalar(err_abs) or np.ndim(err_abs) == 0:
        log_print = '%f' % err_abs
      else:
        log_print = ', '.join('%f' % value for value in err_abs)
      logger.info('err_abs: %s' % log_print)
      logger.info("Pred time: GeoDL: %f s" % (time.time() - start_time))

    if eval_base:
      start_time = time.time()
      self.model.load('')
      y_pred_single = self.model.predict(X)
      
      if MODE == 'classification':
        true_single, total_single, pred_total_single = get_class_wise_accuracy(y, y_pred_single, prf = True)
        pre_single, rec_single, f1_single, total_class = get_prf(true_single, total_single, pred_total_single)
        print('f1_base:', f1_single)
        log_print = ', '.join('%f' % value for value in f1_single)
        logger.info('f1_base: %s' % log_print)
      elif MODE == 'regression':
        y_flat = np.reshape(y, (-1,))
        y_pred_flat = np.reshape(y_pred_single, (-1,))
        err_abs_single = np.sum(np.abs(y_flat - y_pred_flat))
        err_square_single = np.sum(np.square(y_flat - y_pred_flat))
        print('err_abs_base:', err_abs_single)
        print('err_square_base:', err_square_single)
        if np.isscalar(err_abs_single) or np.ndim(err_abs_single) == 0:
          log_print = '%f' % err_abs_single
        else:
          log_print = ', '.join('%f' % value for value in err_abs_single)
        logger.info('err_abs_base: %s' % log_print)
        
      logger.info("Pred time: Base: %f s" % (time.time() - start_time))

    if print_to_file:
      sys.stdout.close()
      sys.stdout = self.original_stdout

    if MODE == 'classification':
      if eval_base:
        return pre, rec, f1, pre_single, rec_single, f1_single
      else:
        return pre, rec, f1
    elif MODE == 'regression':
      if eval_base:
        return err_abs, err_square, err_abs_single, err_square_single
      else:
        return err_abs, err_square

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

    Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)
    results, groups, total_number = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)

    generate_vis_image(self.s_branch, Xtest_branch_id, max_depth = self.max_depth, dir = self.dir_space, step_size = step_size)

    for class_id_input in SELECT_CLASS:
      class_id = int(class_id_input)
      ext = str(class_id)
      grid, vmin, vmax = generate_performance_grid(results, groups, class_id = class_id, step_size = step_size)
      print('X_DIM, grid.shape: ', X_DIM, grid.shape)
      grid_count, vmin_count, vmax_count = generate_count_grid(total_number, groups, class_id = class_id, step_size = step_size)
      generate_vis_image_for_all_groups(grid, dir = self.dir_space, ext = '_star' + ext, vmin = vmin, vmax = vmax)
      generate_vis_image_for_all_groups(grid_count, dir = self.dir_space, ext = '_count' + ext, vmin = vmin_count, vmax = vmax_count)

      results_base, groups_base, _ = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, base = True, X_branch_id = Xtest_branch_id)
      grid_base, vmin_base, vmax_base = generate_performance_grid(results_base, groups_base, class_id = class_id, step_size = step_size)
      generate_vis_image_for_all_groups(grid_base, dir = self.dir_space, ext = '_base' + ext, vmin = vmin_base, vmax = vmax_base)

      cnt_vis_thrd = get_filter_thrd(grid_count, ratio = 0.2)
      grid_diff, vmin_diff, vmax_diff = generate_diff_grid((grid - grid_base)*(grid_count>=cnt_vis_thrd), groups, step_size = step_size)
      generate_vis_image_for_all_groups(grid_diff, dir = self.dir_space, ext = '_diff' + ext, vmin = vmin_diff, vmax = vmax_diff)

      np.save(self.dir_space + '/' + 'grid' + ext + '.npy', grid)
      np.save(self.dir_space + '/' + 'grid_base' + ext + '.npy', grid_base)
      np.save(self.dir_space + '/' + 'grid_count' + ext + '.npy', grid_count)

    return
