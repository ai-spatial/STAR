# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2026-02-07
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


class STARModel:
  """Base class for STAR deep learning models. Subclasses must implement create_net()."""
  def __init__(self, ckpt_path, num_class=NUM_CLASS, mode=MODE, name=None,
               lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
    self.num_class = num_class
    self.mode = mode
    self.ckpt_path = ckpt_path
    self.model_name = name
    self.lr = lr
    self.batch_size = batch_size
    self.epoch_train = epoch_train

  def model_compile(self):
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    if self.mode == 'classification':
      self.model.compile(optimizer=optimizer,
                  loss=CLASSIFICATION_LOSS,
                  metrics=['accuracy'])
    else:
      self.model.compile(optimizer=optimizer,
                  loss=REGRESSION_LOSS,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

  def train(self, X, y, branch_id=None, mode=MODE, train_type='partition'):
    '''
    Input model is compiled!
    Args:
      train_type: 'partition' (default): training during data partitioning; set_trainable_layers is called.
        'plain' (or any string except 'partition'): regular training (e.g. warm-up); set_trainable_layers is not called.
    '''
    if branch_id is None:
      print('Error: branch_id is required for deep learning versions.')
    init_epoch_number = len(branch_id) * self.epoch_train
    if train_type == 'partition':
      self.set_trainable_layers(len(branch_id) + 1)
    self.model.fit(X, y, batch_size=self.batch_size,
              initial_epoch=init_epoch_number, epochs=init_epoch_number + self.epoch_train)
    return

  def predict(self, X, prob=False):
    return self.model.predict(X)

  def predict_test(self, X, y, X_group, s_branch, prf=True, X_branch_id=None, append_acc=False):
    """Branch-wise evaluation."""
    true = 0
    total = 0
    true_class = np.zeros(self.num_class)
    total_class = np.zeros(self.num_class)
    total_pred = np.zeros(self.num_class)
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
        true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf=True)
        true_class += true_class_part
        total_class += total_class_part
        total_pred += total_pred_part
      elif self.mode == 'regression':
        y_part_flat = np.reshape(y_part, (-1,))
        y_pred_flat = np.reshape(y_pred, (-1,))
        err_abs += np.sum(np.abs(y_part_flat - y_pred_flat))
        err_square += np.sum(np.square(y_part_flat - y_pred_flat))
    if self.mode == 'regression':
      return err_abs / y.shape[0], err_square / y.shape[0]
    if prf and self.mode == 'classification':
      if append_acc:
        prf_result = list(get_prf(true_class, total_class, total_pred))
        prf_result.append(np.sum(true) / np.sum(total))
        return tuple(prf_result)
      return get_prf(true_class, total_class, total_pred)
    return true / total

  def predict_geodl(self, X, X_group, s_branch, X_branch_id=None):
    """Predict with branch-specific models, mirroring predict_georf()."""
    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
    y_pred_full = None
    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]
      self.load(branch_id)
      y_pred = self.predict(X_part)
      if y_pred_full is None:
        y_pred_full = np.zeros((X.shape[0],) + y_pred.shape[1:], dtype=y_pred.dtype)
      y_pred_full[id_list] = y_pred
    return y_pred_full

  def set_trainable_layers(self, partition_level, sharing_level = 1.5):
    '''For layer sharing.
    Args:
      partition_level: the level of the partition in the hierarchy.
      sharing_level: controls the number of layers to share after split. When setting to one, no-sharing happens and new models will be trained.
    '''
    total_num_layers = len(self.model.layers)
    num_of_layers_to_train = np.ceil(total_num_layers / sharing_level**partition_level).astype(int)
    num_of_layers_to_train = np.maximum(num_of_layers_to_train, 2)
    print(num_of_layers_to_train)
    for layer in self.model.layers[:-num_of_layers_to_train]:
      layer.trainable = False
    for layer in self.model.layers[-num_of_layers_to_train:]:
      layer.trainable = True
    return

  def save(self, branch_id):
    self.model.save_weights(self.ckpt_path + '/' + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    return

  def load(self, branch_id):
    ckpt_path = self.ckpt_path
    if ckpt_path[-1] != '/':
      ckpt_path = ckpt_path + '/'
    self.model.load_weights(ckpt_path + self.model_name + '_ckpt_' + branch_id + '.weights.h5')
    return


class DNNmodel(STARModel):
  """Dense net implementation. Can be used as a template."""

  def __init__(self, ckpt_path,
               layer_size=INPUT_SIZE, num_layers=NUM_LAYERS, num_class=NUM_CLASS,
               mode=MODE, name=None,
               lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
    super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name,
                     lr=lr, batch_size=batch_size, epoch_train=epoch_train)
    if self.model_name is None:
      self.model_name = 'dnn'
    self.layer_size = layer_size
    self.num_layers = num_layers
    self.model = self.create_net()
    self.model.ckpt_path = ckpt_path
    print('check ckpt path: ' + ckpt_path)

  def create_net(self):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.4)
    inputs = tf.keras.Input(shape=(INPUT_SIZE,))
    x = inputs
    for i in range(self.num_layers):
      x = tf.keras.layers.Dense(self.layer_size, activation=tf.nn.relu, kernel_initializer=initializer)(x)
    final_activation = tf.nn.softmax if self.mode == 'classification' else None
    outputs = tf.keras.layers.Dense(self.num_class, activation=final_activation, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)


class UNetmodel(STARModel):
  """UNet implementation. Can be used as a template."""

  def __init__(self, ckpt_path,
               num_class=NUM_CLASS, mode=MODE, name=None,
               lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN,
               training=True):
    super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name,
                     lr=lr, batch_size=batch_size, epoch_train=epoch_train)
    if self.model_name is None:
      self.model_name = 'unet'
    self.model = self.create_net(training=training)
    self.model.ckpt_path = ckpt_path
    print('check ckpt path: ' + ckpt_path)

  def create_net(self, training=True):
    '''
    Args:
      training: set to False for inference mode, which is mainly used to freeze batch norms.
        Use True when performing initial training on the base.
    '''
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, INPUT_SIZE])
    x = inputs
    down_stack = [
        conv(16, 3, apply_norm=True),
        conv(32, 3, apply_norm=True),
        conv(64, 3, apply_norm=True)
        ]
    up_stack = [
        upsample(128, 3, apply_norm=True),
        upsample(64, 3, apply_norm=True),
        upsample(32, 3, apply_norm=True)
        ]
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        x = maxpool(x)
    skips = reversed(skips)
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])
    x = conv(32, 3, apply_norm=True)(x)
    last = tf.keras.layers.Conv2D(self.num_class, (3, 3), strides=1, padding='same')
    x = last(x)
    output = tf.nn.softmax(x)
    net = tf.keras.Model(inputs=inputs, outputs=output)
    output_final = net(inputs, training=training)
    return tf.keras.Model(inputs=inputs, outputs=output_final, name=self.model_name)


class LSTMmodel(STARModel):
  """LSTM on sequences of shape (N_TIME, INPUT_SIZE): dense embedding per timestep, then LSTM (one output per sequence)."""

  def __init__(self, ckpt_path,
               n_time=N_TIME, layer_size=INPUT_SIZE, num_layers=NUM_LAYERS, num_class=NUM_CLASS,
               mode=MODE, name=None,
               lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
    super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name,
                     lr=lr, batch_size=batch_size, epoch_train=epoch_train)
    if self.model_name is None:
      self.model_name = 'lstm'
    self.n_time = n_time
    self.layer_size = layer_size
    self.num_layers = num_layers
    self.model = self.create_net()
    self.model.ckpt_path = ckpt_path
    print('check ckpt path: ' + ckpt_path)

  def create_net(self):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.4)
    # Input: (N_TIME, INPUT_SIZE)
    inputs = tf.keras.Input(shape=(self.n_time, INPUT_SIZE))
    x = inputs
    # Dense embedding applied per timestep (same weights for all timesteps)
    for i in range(self.num_layers):
      x = tf.keras.layers.TimeDistributed(
          tf.keras.layers.Dense(self.layer_size, activation=tf.nn.relu, kernel_initializer=initializer)
      )(x)
    # LSTM: one vector per sequence (return_sequences=False)
    x = tf.keras.layers.LSTM(
        self.layer_size, activation=tf.nn.tanh,
        kernel_initializer=initializer, return_sequences=False
    )(x)
    final_activation = tf.nn.softmax if self.mode == 'classification' else None
    outputs = tf.keras.layers.Dense(
        self.num_class, activation=final_activation, kernel_initializer=initializer
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)

'''Helper functions for models.'''
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
