# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-11-30
# @License: MIT License

import numpy as np
import tensorflow as tf
import pandas as pd

from config import *
from helper import get_X_branch_id_by_group
from metrics import *

NUM_LAYERS = 8 #num_layers is the number of non-input layers

'''Model can be easily customized to different deep network architectures.
The following key functions need to be included (just adding some steps around regular tf functions as needed), which are used in STAR training:
1. train()
2. model_complie()
3. predict(): this is the regular prediction function from X->y (using a single branch), which returns predicted labels
4. save(): save a branch
5. load(): load a branch
6. set_trainable_layers(): used to freeze weights on layers that are not used to construct new branches at a split

Optional (not called during STAR training, used either in model init() or final test):
7. create_net(): used to construct the network.
8. predict_test(): this is for STAR's prediction, which make predictions based on which branch a sample belongs to, and returns performance metric values (can change)
'''

'''This is an example implementation for a dense net. Can be used as a template.'''
class DNNmodel():#tf.keras.Model

  def __init__(self, ckpt_path,
               layer_size = INPUT_SIZE, num_layers = NUM_LAYERS, num_class = NUM_CLASS,
               mode = MODE, name = None,
               lr = LEARNING_RATE, batch_size = BATCH_SIZE, epoch_train = EPOCH_TRAIN):
    # super(DenseNet, self).__init__(name = name)

    #inputs (definition)
    self.layer_size = layer_size
    self.num_layers = num_layers#not including the output layer
    self.num_class = num_class
    self.mode = mode
    self.ckpt_path = ckpt_path
    if name is None:
      self.model_name = 'dnn'
    else:
      self.model_name = name

    #define model
    self.model = self.create_net()
    self.model.ckpt_path = ckpt_path
    print('check ckpt path: ' + ckpt_path)

    #training related
    self.lr = lr
    self.batch_size = batch_size
    self.epoch_train = epoch_train#this is #epochs to train after each split

  def create_net(self):

    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.4)#mean=0.0, seed=None

    inputs = tf.keras.Input(shape=(INPUT_SIZE,))
    x = inputs
    for i in range(self.num_layers):
      x = tf.keras.layers.Dense(self.layer_size, activation=tf.nn.relu, kernel_initializer=initializer)(x)

    final_activation = None
    if self.mode == 'classification':
      final_activation = tf.nn.softmax
    # else:
    #   final_activation = tf.nn.relu

    outputs = tf.keras.layers.Dense(self.num_class, activation=final_activation, kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)

  # https://www.tensorflow.org/guide/keras/train_and_evaluate
  def model_compile(self):

    optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

    if self.mode == 'classification':
      self.model.compile(optimizer=optimizer,
                  loss=CLASSIFICATION_LOSS,
                  metrics=['accuracy'])
    else:
      self.model.compile(optimizer=optimizer,
                  loss=REGRESSION_LOSS,
                  metrics=[tf.keras.metrics.MeanSquaredError()])


  def train(self, X, y, branch_id = None, mode = MODE, train_type = 'partition'):
    '''
    Input model is complied!
    Args:
      train_type: takes two values:
        'partition': default value. This means the training is called during the data partitioning process.
            In this case, the set_trainable_layers function will be called.
        'plain': This means a regular training (e.g., warm-up training before partitioning starts).
            The set_trainable_layers function will not be called.
    '''

    if branch_id is None:
      print('Error: branch_id is required for deep learning versions.')

    # lr_scheduler = LearningRateScheduler(decay_schedule)
    # callbacks_all = [lr_scheduler]

    # if callbacks is not None:
    #   callbacks_all.extend(callbacks)
    init_epoch_number = len(branch_id) * self.epoch_train

    if train_type == 'partition':
      self.set_trainable_layers(len(branch_id)+1)

    #train the model for epoch_train number of epochs
    self.model.fit(X, y, batch_size = self.batch_size,
              initial_epoch=init_epoch_number, epochs=init_epoch_number + self.epoch_train)#callbacks=callbacks_all, init_epoch_number +
    return

  def predict(self, X, prob = False):
    #no change
    return self.model.predict(X)

  def predict_test(self, X, y, X_group, s_branch, prf = True, X_branch_id = None, append_acc = False):
    """
    Branch-wise evaluation, mirroring RFmodel.predict_test().
    """
    '''Classification'''
    true = 0
    total = 0
    true_class = np.zeros(self.num_class)
    total_class = np.zeros(self.num_class)
    total_pred =  np.zeros(self.num_class)
    '''Regression'''
    err_abs = 0
    err_square = 0

    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]
      y_part = y[id_list]

      self.load(branch_id)
      y_pred = self.predict(X_part)

      if self.mode == 'classification':
        true_part, total_part = get_overall_accuracy(y_part, y_pred)
        true += true_part
        total += total_part

        true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf = True)
        true_class += true_class_part
        total_class += total_class_part
        total_pred += total_pred_part

      if self.mode == 'regression':
        y_part_flat = np.reshape(y_part, (-1,))
        y_pred_flat = np.reshape(y_pred, (-1,))
        err_abs_part = np.sum(np.abs(y_part_flat - y_pred_flat))
        err_square_part = np.sum(np.square(y_part_flat - y_pred_flat))
        err_abs += err_abs_part
        err_square += err_square_part

    
    if self.mode == 'regression':
      return err_abs / y.shape[0], err_square / y.shape[0]

    if prf and self.mode == 'classification':
      if append_acc:
        prf_result = list(get_prf(true_class, total_class, total_pred))
        prf_result.append(np.sum(true) / np.sum(total))
        return tuple(prf_result)
      return get_prf(true_class, total_class, total_pred)

    return true / total

  # def get_score(self, y_true, y_pred_prob):
  #   if self.mode == 'classification':
  #     y_pred = np.argmax(y_pred_prob, axis=1)
  #     if len(y_true.shape) > 1:
  #       y_true = np.argmax(y_true, axis=1)
  #     return np.mean(y_pred == y_true)
  #   return None

  def predict_geodl(self, X, X_group, s_branch, X_branch_id = None):
    """
    Predict with branch-specific models, mirroring predict_georf().
    """
    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    y_pred_full = None
    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]

      self.load(branch_id)
      y_pred = self.predict(X_part)

      if y_pred_full is None:
        y_pred_full = np.zeros((X.shape[0],) + y_pred.shape[1:], dtype = y_pred.dtype)
      y_pred_full[id_list] = y_pred

    return y_pred_full

  #tensorflow.org/guide/keras/sequential_model
  def set_trainable_layers(self, partition_level):

    # num_of_layers_to_train = np.max([1, np.floor(NUM_LAYERS_DNN / 2**partition_level)]).astype(int)
    total_num_layers = len(self.model.layers)
    num_of_layers_to_train = np.ceil(total_num_layers / 1.5**partition_level).astype(int)
    num_of_layers_to_train = np.maximum(num_of_layers_to_train, 2)
    print(num_of_layers_to_train)
    for layer in self.model.layers[:-num_of_layers_to_train]:
      layer.trainable = False

    for layer in self.model.layers[-num_of_layers_to_train:]:
      layer.trainable = True

    return

  '''The save and load functions will be used during partitioning optimization and training.'''
  def save(self, branch_id):
    #save the current branch
    #branch_id should include the current branch (not after added to X_branch_id)
    self.model.save_weights(self.ckpt_path + '/' + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    # print(self.ckpt_path + 'ckpt_' + self.model_name + '_' + branch_id)
    return

  def load(self, branch_id):
    #load the base branch before further fine-tuning
    ckpt_path = self.ckpt_path
    if ckpt_path[-1] != '/':
      ckpt_path = ckpt_path + '/'
    self.model.load_weights(ckpt_path + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    # print(self.ckpt_path + 'ckpt_' + self.model_name + '_' + branch_id)
    return


'''This is an example implementation for UNet. Can be used as a template.'''
class UNetmodel():#tf.keras.Model

  def __init__(self, ckpt_path,
               num_class = NUM_CLASS,
               mode = MODE, name = None,
               lr = LEARNING_RATE, batch_size = BATCH_SIZE, epoch_train = EPOCH_TRAIN,
               training = True):
               # layer_size = INPUT_SIZE, num_layers = NUM_LAYERS,
    # super(DenseNet, self).__init__(name = name)

    #inputs (definition)
    # self.layer_size = layer_size
    # self.num_layers = num_layers#not including the output layer
    self.num_class = num_class
    self.mode = mode
    self.ckpt_path = ckpt_path
    if name is None:
      self.model_name = 'unet'
    else:
      self.model_name = name

    #define model
    self.model = self.create_net(training = training)
    self.model.ckpt_path = ckpt_path
    print('check ckpt path: ' + ckpt_path)

    #training related
    self.lr = lr
    self.batch_size = batch_size
    self.epoch_train = epoch_train#this is #epochs to train after each split

  def create_net(self, training = True):
    '''
    Args:
      training: set to False for inference mode, which is mainly used to freeze batch norms.
        Use True when performing initial training on the base.
    '''

    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, INPUT_SIZE])
    x = inputs

    down_stack = [
        conv( 16, 3, apply_norm=True), # 128x128 -> (bs, 64, 64, 32)
        conv( 32, 3, apply_norm=True), # 64x64 -> (bs, 32, 32, 64)
        conv( 64, 3, apply_norm=True) # 32x32 -> (bs, 16, 16, 128)
        # conv(128, 3, apply_norm=True), # 16x16 -> (bs,  8,  8, 256)
        ]

    up_stack = [
        upsample(128, 3, apply_norm=True),#   8x8 -> (bs,  16,  16, 256)
        upsample(64, 3, apply_norm=True),# 16x16 -> (bs,  32,  32, 128)
        upsample(32, 3, apply_norm=True)# 32x32 -> (bs,  64,  64, 64)
        # upsample(32, 3, apply_norm=True) # 64x64 -> (bs, 128, 128, 32)
        ]

    # Downsampling through the model
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        x = maxpool(x)

    # we take reverse
    skips = reversed(skips)

    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    x = conv(32, 3, apply_norm=True)(x)
    last = tf.keras.layers.Conv2D(self.num_class, (3,3), strides=1, padding='same')#, activation='sigmoid', use_bias=False
    x = last(x)
    output = tf.nn.softmax(x)

    net = tf.keras.Model(inputs=inputs, outputs=output)

    #add flexibility to switch between training and inference
    output_final = net(inputs, training = training)

    return tf.keras.Model(inputs=inputs, outputs=output_final, name=self.model_name)

  # https://www.tensorflow.org/guide/keras/train_and_evaluate
  def model_compile(self):

    optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

    if self.mode == 'classification':
      self.model.compile(optimizer=optimizer,
                  loss=CLASSIFICATION_LOSS,
                  metrics=['accuracy'])
    else:
      self.model.compile(optimizer=optimizer,
                  loss=REGRESSION_LOSS,
                  metrics=[tf.keras.metrics.MeanSquaredError()])


  def train(self, X, y, branch_id = None, mode = MODE, train_type = 'partition'):
    '''
    Input model is complied!
    Args:
      train_type: takes two values:
        'partition': default value. This means the training is called during the data partitioning process.
            In this case, the set_trainable_layers function will be called.
        'plain' (or any string except 'partition'): This means a regular training (e.g., warm-up training before partitioning starts).
            The set_trainable_layers function will not be called.
    '''

    if branch_id is None:
      print('Error: branch_id is required for deep learning versions.')

    # lr_scheduler = LearningRateScheduler(decay_schedule)
    # callbacks_all = [lr_scheduler]

    # if callbacks is not None:
    #   callbacks_all.extend(callbacks)
    init_epoch_number = len(branch_id) * self.epoch_train

    if train_type == 'partition':
      self.set_trainable_layers(len(branch_id)+1)

    #train the model for epoch_train number of epochs
    self.model.fit(X, y, batch_size = self.batch_size,
              initial_epoch=init_epoch_number, epochs=init_epoch_number + self.epoch_train)#callbacks=callbacks_all, init_epoch_number +, training = training
    return

  def predict(self, X, prob = False):
    #no change
    return self.model.predict(X)

  def predict_test(self, X, y, X_group, s_branch, prf = True, X_branch_id = None, append_acc = False):
    """
    Branch-wise evaluation, mirroring RFmodel.predict_test().
    """
    '''Classification'''
    true = 0
    total = 0
    true_class = np.zeros(self.num_class)
    total_class = np.zeros(self.num_class)
    total_pred =  np.zeros(self.num_class)
    '''Regression'''
    err_abs = 0
    err_square = 0

    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]
      y_part = y[id_list]

      self.load(branch_id)
      y_pred = self.predict(X_part)

      if self.mode == 'classification':
        true_part, total_part = get_overall_accuracy(y_part, y_pred)
        true += true_part
        total += total_part

        true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf = True)
        true_class += true_class_part
        total_class += total_class_part
        total_pred += total_pred_part

      elif self.mode == 'regression':
        y_part_flat = np.reshape(y_part, (-1,))
        y_pred_flat = np.reshape(y_pred, (-1,))
        err_abs_part = np.sum(np.abs(y_part_flat - y_pred_flat))
        err_square_part = np.sum(np.square(y_part_flat - y_pred_flat))
        err_abs += err_abs_part
        err_square += err_square_part

    if prf and self.mode == 'classification':
      if append_acc:
        prf_result = list(get_prf(true_class, total_class, total_pred))
        prf_result.append(np.sum(true) / np.sum(total))
        return tuple(prf_result)
      return get_prf(true_class, total_class, total_pred)

    if self.mode == 'regression':
      return err_abs / y.shape[0], err_square / y.shape[0]
    
    return true / total

  # def get_score(self, y_true, y_pred_prob):
  #   if self.mode == 'classification':
  #     y_pred = np.argmax(y_pred_prob, axis=1)
  #     if len(y_true.shape) > 1:
  #       y_true = np.argmax(y_true, axis=1)
  #     return np.mean(y_pred == y_true)
  #   return None

  def predict_geodl(self, X, X_group, s_branch, X_branch_id = None):
    """
    Predict with branch-specific models, mirroring predict_georf().
    """
    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    y_pred_full = None
    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]

      self.load(branch_id)
      y_pred = self.predict(X_part)

      if y_pred_full is None:
        y_pred_full = np.zeros((X.shape[0],) + y_pred.shape[1:], dtype = y_pred.dtype)
      y_pred_full[id_list] = y_pred

    return y_pred_full

  #tensorflow.org/guide/keras/sequential_model
  def set_trainable_layers(self, partition_level):

    # num_of_layers_to_train = np.max([1, np.floor(NUM_LAYERS_DNN / 2**partition_level)]).astype(int)
    total_num_layers = len(self.model.layers)
    num_of_layers_to_train = np.ceil(total_num_layers / 1.5**partition_level).astype(int)
    num_of_layers_to_train = np.maximum(num_of_layers_to_train, 2)
    print(num_of_layers_to_train)
    for layer in self.model.layers[:-num_of_layers_to_train]:
      layer.trainable = False

    for layer in self.model.layers[-num_of_layers_to_train:]:
      layer.trainable = True

    return

  '''The save and load functions will be used during partitioning optimization and training.'''
  def save(self, branch_id):
    #save the current branch
    #branch_id should include the current branch (not after added to X_branch_id)
    self.model.save_weights(self.ckpt_path + '/' + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    # print(self.ckpt_path + 'ckpt_' + self.model_name + '_' + branch_id)
    return

  def load(self, branch_id):
    #load the base branch before further fine-tuning
    ckpt_path = self.ckpt_path
    if ckpt_path[-1] != '/':
      ckpt_path = ckpt_path + '/'
    self.model.load_weights(ckpt_path + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    # print(self.ckpt_path + 'ckpt_' + self.model_name + '_' + branch_id)
    return

#removing redundancy in examples
# '''General model functions.
# Used for convenience in model definition,
# or outside model training or data partitioning.
# Includes branch-wise prediction helpers for STAR.'''
# def predict_test(model, X, y, X_group, s_branch, prf = True):
#   #prob here is aggregated probability (does not sum to 1 without normalizing)
#   '''
#   group_branch contains the branch_id for each group
#   '''
#   true = 0
#   total = 0
#   true_class = np.zeros(model.num_class)
#   total_class = np.zeros(model.num_class)
#   total_pred =  np.zeros(model.num_class)

#   X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
#   for branch_id in np.unique(X_branch_id):
#     id_list = np.where(X_branch_id == branch_id)
#     X_part = X[id_list]
#     y_part = y[id_list]

#     model.load(branch_id)
#     # y_pred = model.predict(X_part)

#     if model.mode == 'classification':
#       true_part, total_part, true_class_part, total_class_part, total_pred_part = predict_test_one_branch(model, X, y, as_sub_function = True)
#       #overall
#       # true_part, total_part = get_overall_accuracy(y_part, y_pred)
#       true += true_part
#       total += total_part
#       #class-wise, if needed
#       # true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf = True)
#       true_class += true_class_part
#       total_class += total_class_part
#       total_pred += total_pred_part

#   if prf:
#     return get_prf(true_class, total_class, total_pred)#acc, acc_class
#   else:
#     return true / total

# def predict_test_one_branch(model, X, y, prf = True, batch_size = 32, as_sub_function = False):
#   '''Used to perform testing using a single branch (e.g., base network with branch_id = '').
#   Args:
#     as_sub_function: if True, return intermediate results to use in other functions.
#   '''
#   # model.load('')#'' is the base branch

#   true = 0
#   total = 0
#   true_class = np.zeros(model.num_class)
#   total_class = np.zeros(model.num_class)
#   total_pred =  np.zeros(model.num_class)

#   for i in range(0, X.shape[0], batch_size):
#     i1 = min(i+batch_size, X.shape[0])

#     X_part = X[i:i1]
#     y_part = y[i:i1]

#     y_pred = model.predict(X_part)

#     if model.mode == 'classification':
#       #overall
#       true_part, total_part = get_overall_accuracy(y_part, y_pred)
#       true += true_part
#       total += total_part

#       #class-wise, if needed
#       true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf = True)
#       true_class += true_class_part
#       total_class += total_class_part
#       total_pred += total_pred_part

#   if as_sub_function:
#     return true, total, true_class, total_class, total_pred
#   else:
#     if prf:
#       return get_prf(true_class, total_class, total_pred)#acc, acc_class
#     else:
#       return true / total

#   # y_pred = model.predict(X)
#   #
#   # if model.mode == 'classification':
#   #   #overall
#   #   true, total = get_overall_accuracy(y, y_pred)
#   #   #class-wise, if needed
#   #   true_class, total_class, total_pred = get_class_wise_accuracy(y, y_pred, prf = True)
#   #
#   # if prf:
#   #   return get_prf(true_class, total_class, total_pred)#acc, acc_class
#   # else:
#   #   return true / total


def save_single(model, path, name = 'single'):
  """
  Save a single model checkpoint. For deep models, use model.save() when available.
  """
  if hasattr(model, 'save'):
    model.save(name)
    return
  if hasattr(model, 'model'):
    model.model.save_weights(path + '/' + name)


def conv(filters, size, norm_type='batchnorm', apply_norm=True):
    # from pix2pix
    """Downsamples an input.
    Conv2D+ReLU (+BN) => Conv2D+ReLU (+BN) => MaxPool
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_norm: If True, adds the batchnorm layer
    Returns:
      Downsample Sequential Model
    """
    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    # First Conv2D
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=None,
                                      use_bias=True))
                                      #kernel_regularizer=regularizers.l2(0)
    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
      # elif norm_type.lower() == 'instancenorm':
      #   result.add(InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    # result.add(tf.keras.layers.Dropout(0.1))

    # Second Conv2D
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=None,
                                      use_bias=True))

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
      # elif norm_type.lower() == 'instancenorm':
      #   result.add(InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    # result.add(tf.keras.layers.Dropout(0.1))

    return result

def upsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """Upsamples an input.
    Conv2D+ReLU (+BN) => Conv2D+ReLU (+BN) => Conv2DTranspose
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """

    # TBD - add L2 regularization in conv2d or dropout for reducing overfitting

    result = tf.keras.Sequential()

    # First Conv2D
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=None,
                                      use_bias=True))

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
      # elif norm_type.lower() == 'instancenorm':
      #   result.add(InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    # result.add(tf.keras.layers.Dropout(0.1))

    # Second Conv2D
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation='relu',
                                      use_bias=True))

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
      # elif norm_type.lower() == 'instancenorm':
      #   result.add(InstanceNormalization())

    # result.add(tf.keras.layers.Dropout(0.1))

    # Upsample
    # note filter size is reduced /2 since it's concatenated
    result.add(tf.keras.layers.Conv2DTranspose(int(filters/2), size, strides=2, padding='same',
                                               use_bias=True))

    return result
