# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

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
