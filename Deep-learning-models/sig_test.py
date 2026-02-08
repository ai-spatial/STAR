# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from config import *
from static_table import cvtable

#significance testing functions

def sig_test(base_score, split_score0, split_score1, base_score_before = None):#diff
  '''Significance testing: pure numerical-based test (spatial test is V2 for arbitrary subset of grid cells).
  This function needs cv_table to be loaded first (available in main.py).
  '''

  #prepare diff
  split_score = np.hstack([split_score0, split_score1])
  diff = split_score - base_score

  print("mean split score: %f" % (np.mean(split_score)))
  print("mean base score: %f" % (np.mean(base_score)))

  #testing related paras
  sig = 0
  degree_freedom = diff.shape[0]

  mean_diff = np.mean(diff)

  #may require sufficient #epochs to avoid random results before converging/stablized
  if mean_diff <= 0:#no improvement
      return 0

  std_diff = np.std(diff)
  # std_diff = max(1, std_diff)
  std_err_mean = std_diff/np.sqrt(degree_freedom+1)

  test_stat = mean_diff/std_err_mean#interpretation: (mean_diff-0)/std_err_mean

  #compare with critical values to determine if significant
  cvtable_col = 1
  if SIGLVL == 0.05:
      cvtable_col = 1
  elif SIGLVL == 0.01:
      cvtable_col = 2
  elif SIGLVL == 0.005:
      cvtable_col = 3
  elif SIGLVL == 0.001:
      cvtable_col = 4
  elif SIGLVL == 0.0005:
      cvtable_col = 5
  elif SIGLVL == 0.1:
      cvtable_col = 6
  else:
      cvtable_col = 1#add conditions if more sig levels are needed (in case 0.05 is too strict)

  #find nearest df value
  nn_idx = np.abs(cvtable[:,0] - degree_freedom).argmin()
  nn_idx_other = -1
  if cvtable[nn_idx, 0] >= degree_freedom and degree_freedom>=1:
      nn_idx_other = nn_idx - 1#assumes df>=1 (using #validation samples as a condition for branching)
  else:
      nn_idx_other = nn_idx + 1

  #approximate cv value for current partition
  df_cv_a = cvtable[nn_idx, 0]
  df_cv_b = cvtable[nn_idx_other, 0]

  cv_a = cvtable[nn_idx, cvtable_col]
  cv_b = cvtable[nn_idx_other, cvtable_col]

  cv_apx = (1/degree_freedom - 1/df_cv_a) / (1/df_cv_b - 1/df_cv_a) * (cv_b - cv_a) + cv_a

  '''Effect size.'''
  es = 0

  #def 1: regular
  # es = mean_diff / std_diff

  #def 2: split diff std
  # split0_size = split_score0.shape[0]
  # diff0 = split_score0 - base_score[:split0_size]
  # diff1 = split_score1 - base_score[split0_size:]
  # base_size = base_score.shape[0]
  # split0_ratio = split0_size / base_size
  # split_std_diff = split0_ratio * np.std(diff0) + (1 - split0_ratio) * np.std(diff1)
  # es = mean_diff / split_std_diff

  #def 3: score std
  std_score = np.std(split_score)
  es = mean_diff/ (0.5 * std_score + 0.5 * std_diff)

  #def 4: using y_pred_base before training
  if base_score_before is not None:
    diff_base_before = base_score - base_score_before
    mean_diff_base_before = np.mean(diff_base_before)
    es = mean_diff / mean_diff_base_before

    if mean_diff > 0 and mean_diff_base_before < 0:
      es = 100

  print("mean_diff: %f" % (mean_diff))
  print("mean_diff_base_before: %f" % (mean_diff_base_before))
  # print("mean_diff - mean_diff_base_before: %f" % (mean_diff - mean_diff_base_before))
  print("es: %f" % (es))

  print("mean_diff: {:.3f}".format(mean_diff) + ", std_diff: {:.3f}".format(std_diff) + ", std_score: {:.3f}".format(std_score) + ", es: {:.3f}".format(es))
  # print("mean_diff: {:.3f}".format(mean_diff) + ", std_diff: {:.3f}".format(std_diff) + ", split_std_diff: {:.3f}".format(split_std_diff) + ", es: {:.3f}".format(es))
  print("df: {:.2f}".format(degree_freedom) + ", test_stat: {:.3f}".format(test_stat) + ", cv_apx: " +  "{:.3f}".format(cv_apx))

  #test significance under siglvl
  if test_stat >= cv_apx and es >= ES_THRD and mean_diff >= MD_THRD:
      sig = 1

  return sig
