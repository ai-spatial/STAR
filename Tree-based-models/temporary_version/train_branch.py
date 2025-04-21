# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import models#tuning code

def base_eval_using_merged_branch_data(model, X_val, branch_id):
  '''
  This evaluates the base scenario before training is carried out (to compare split vs. no split).
  Makes sure the order of samples remains the same for significance testing.
  '''
  model.load(branch_id)
  y_pred = model.predict(X_val)
  return y_pred

def train_and_eval_using_merged_branch_data(model, X_train, y_train, X_val,
                                            branch_id):
  '''
  This is for the scenario with no split.
  Makes sure the order of samples remains the same for significance testing
  '''
  model.load(branch_id)
  model.train(X_train, y_train, branch_id)
  model.save(branch_id)
  y_pred = model.predict(X_val)
  return y_pred

def train_and_eval_two_branch(model, X0_train, y0_train, X0_val,
                              X1_train, y1_train, X1_val,
                              branch_id):
  '''
  This is for the scenario with split.
  '''

  #branch 0
  print("Training branch 0:")
  model.load(branch_id)
  model.train(X0_train, y0_train, branch_id)
  y0_pred = model.predict(X0_val)
  model.save(branch_id + '0')

  #branch 1
  print("Training branch 1:")
  model.load(branch_id)
  model.train(X1_train, y1_train, branch_id)
  y1_pred = model.predict(X1_val)
  model.save(branch_id + '1')


  return y0_pred, y1_pred



# def eval_and_select_model_RF(model, X0_val, y0_val, X1_val, y1_val, split_score0, split_score1):
#   '''For RF, one of the two partitions may not necessarily benefit from the data reduction based on heterogeneity.
#   For example, a large branch may not be impacted by small branch data. Adding more data may improve independence between trees.'''
#
#   y0_pred_before = base_eval_using_merged_branch_data(model, X0_val, branch_id)
#   y1_pred_before = base_eval_using_merged_branch_data(model, X1_val, branch_id)
#   split_score0_before, split_score1_before = get_split_score(y0_val, y0_pred_before, y1_val, y1_pred_before)
#   print('effects of using only data in partition:')
#   print('score before 0: ', np.mean(split_score0_before), 'score after 0: ', np.mean(split_score0))
#   print('score before 1: ', np.mean(split_score1_before), 'score after 1: ', np.mean(split_score1))
#   if np.mean(split_score0_before) >= np.mean(split_score0) and np.mean(split_score1_before) < np.mean(split_score1):
#     #overwrite
#     split_score0 = split_score0_before
#     model.load(branch_id)
#     model.save(branch_id + '0')
#     print('overwrite branch', branch_id + '0', ' weights with branch', branch_id)
#
#   elif np.mean(split_score0_before) < np.mean(split_score0) and np.mean(split_score1_before) >= np.mean(split_score1):
#     #overwrite
#     split_score1 = split_score1_before
#     model.load(branch_id)
#     model.save(branch_id + '1')
#     print('overwrite branch', branch_id + '1', ' weights with branch', branch_id)


def decay_schedule(epoch, lr):
  if (epoch % 10 == 0) and (epoch != 0):
  # if (epoch % 50 == 0) and (epoch != 0):
      lr = 0.01
      # lr = lr * 0.95
      print("learning rate: ", lr)
  return lr

def save_single(model, path, name = 'single'):
    #only saves the current new forest (newly added one)
    filename = 'rf_' + name
    pickle.dump(model, open(path + '/' + filename, 'wb'))
