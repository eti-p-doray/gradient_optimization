from datetime import datetime
import argparse
import math
import time
from itertools import chain

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import wandb

from optimizers import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
  tf.config.experimental.set_memory_growth(gpu, True)

def compute_fans(shape):
  """Computes the number of input and output units for a weight shape.
  Args:
    shape: Integer shape tuple or TF tensor shape.
  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)

class MLPModel(tf.keras.Model):
  def __init__(self, train_step_fn, sample_loss_hessian_fn):
    super(MLPModel, self).__init__()
    self.train_step_fn = train_step_fn
    self.sample_loss_hessian = sample_loss_hessian_fn

    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(200, activation='relu')
    self.dense2 = tf.keras.layers.Dense(200, activation='relu')
    self.dense3 = tf.keras.layers.Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  def sample_information(self, x, sample_fn):
    with tf.GradientTape() as tape:
      y_pred = self.call(x, training=True)
    sample = self.sample_loss_hessian(y_pred, sample_fn(y_pred.shape))
    fisher_sample = tape.gradient(y_pred, self.trainable_weights, output_gradients=sample)
    information_scalars = [tf.zeros([1]) for weight in self.trainable_weights]
    return fisher_sample, information_scalars

  def call(self, x, training=False):
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

  def train_step(self, data):
    return self.train_step_fn(self, data)

class AutoEncodeMNIST(tf.keras.Model):
  def __init__(self, train_step_fn, sample_loss_hessian_fn):
    super(AutoEncodeMNIST, self).__init__()
    self.train_step_fn = train_step_fn
    self.sample_loss_hessian = sample_loss_hessian_fn
  
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64, activation='relu')
    self.dense3 = tf.keras.layers.Dense(32, activation='relu')

    self.dense4 = tf.keras.layers.Dense(64, activation='relu')
    self.dense5 = tf.keras.layers.Dense(128, activation='relu')
    self.dense6 = tf.keras.layers.Dense(784, activation='sigmoid')

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 100])

  def get_config(self):
    return {}

  def call(self, x):
    x = self.flatten(x)

    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)

    x = self.dense4(x)
    x = self.dense5(x)
    return self.dense6(x)

  def sample_observations(self, observations, sample):
    return self.sample_loss_hessian(observations, sample)

  def fim_diagonal(self, x):
    return [tf.zeros([1]) for weight in self.trainable_weights]

  def trainable_weight_initializers(self):
    for weight in self.trainable_weights:
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast((fan_in + fan_out) / 2.0, tf.float32))

  def train_step(self, data):
    return self.train_step_fn(self, data)

class CNNModel(tf.keras.Model):
  def __init__(self, train_step_fn, sample_loss_hessian_fn):
    super(CNNModel, self).__init__()
    self.train_step_fn = train_step_fn
    self.sample_loss_hessian = sample_loss_hessian_fn

    self.flatten = tf.keras.layers.Flatten()
    self.conv1 = tf.keras.layers.Conv2D(32, 5, activation='relu')
    self.pool1 = tf.keras.layers.MaxPool2D()
    self.conv2 = tf.keras.layers.Conv2D(64, 5, activation='relu')
    self.pool2 = tf.keras.layers.MaxPool2D()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  def call(self, x, training=False):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

  def sample_observations(self, observations, sample):
    return self.sample_loss_hessian(observations, sample)

  def fim_diagonal(self, x):
    return [tf.zeros([1]) for weight in self.trainable_weights]

  def trainable_weight_initializers(self):
    for weight in self.trainable_weights:
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast(fan_in, tf.float32))

  def train_step(self, data):
    return self.train_step_fn(self, data)

class VGGModel(tf.keras.Model):
  def __init__(self, train_step_fn, sample_loss_hessian_fn):
    super(VGGModel, self).__init__()
    self.train_step_fn = train_step_fn
    self.sample_loss_hessian = sample_loss_hessian_fn

    self.conv1_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.conv1_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.pool1 = tf.keras.layers.MaxPool2D()
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.conv2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.conv2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.pool2 = tf.keras.layers.MaxPool2D()
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    self.conv3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.conv3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    self.pool3 = tf.keras.layers.MaxPool2D()
    self.dropout3 = tf.keras.layers.Dropout(0.2)
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128)
    self.dropout4 = tf.keras.layers.Dropout(0.2)
    self.d2 = tf.keras.layers.Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  def call(self, x, training=False):
    x = self.conv1_1(x)
    x = self.conv1_2(x)
    x = self.pool1(x)
    x = self.dropout1(x, training=training)

    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.pool2(x)
    x = self.dropout2(x, training=training)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.pool3(x)
    x = self.dropout3(x, training=training)

    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout4(x, training=training)
    return self.d2(x)

  def sample_observations(self, observations, sample):
    return self.sample_loss_hessian(observations, sample)

  def fim_diagonal(self, x):
    return [tf.zeros([1]) for weight in self.trainable_weights]

  def trainable_weight_initializers(self):
    for weight in self.trainable_weights:
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast(fan_in, tf.float32))

  def train_step(self, data):
    return self.train_step_fn(self, data)


class RNNModel_IMDB(tf.keras.Model):
  def __init__(self, encoder, train_step_fn, sample_loss_hessian_fn):
    super(RNNModel_IMDB, self).__init__()
    self.train_step_fn = train_step_fn
    self.sample_loss_hessian = sample_loss_hessian_fn

    self.encoder = encoder
    self.embedding = tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True, 
          embeddings_initializer=tf.keras.initializers.VarianceScaling(mode='fan_out'))
    self.dropout1 = tf.keras.layers.Dropout(0.5)
    self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, recurrent_initializer='glorot_uniform'))
    self.dropout2 = tf.keras.layers.Dropout(0.4)
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dropout3 = tf.keras.layers.Dropout(0.2)
    self.dense2 = tf.keras.layers.Dense(1)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0]])

  def get_config(self):
    return {}

  def call(self, x, training=False):
    x = self.encoder(x)
    x = self.embedding(x)
    x = self.dropout1(x, training=training)
    x = self.lstm(x, training=training)
    x = self.dropout2(x, training=training)
    x = self.dense1(x)
    x = self.dropout3(x, training=training)
    return self.dense2(x)

  def sample_observations(self, observations, sample):
    return self.sample_loss_hessian(observations, sample)

  def fim_diagonal(self, x):
    return [tf.zeros([1]) for weight in self.trainable_weights]

  def trainable_weight_initializers(self):
    for weight in self.embedding.trainable_weights:
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast(fan_out, tf.float32))
    for weight in self.lstm.trainable_weights:
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast((fan_in + fan_out) / 2.0, tf.float32))
    for weight in chain(self.dense1.trainable_weights, self.dense2.trainable_weights):
      fan_in, fan_out = compute_fans(weight.shape)
      yield (weight, tf.cast((fan_in + fan_out) / 2.0, tf.float32))

  def train_step(self, data):
    return self.train_step_fn(self, data)

def sample_crossentropy_hessian(predictions, samples):
  y = tf.nn.softmax(predictions)
  z = tf.sqrt(y)
  return z * samples - y * tf.reduce_sum(z * samples, axis=-1, keepdims=True)

def sample_binary_crossentropy_hessian(predictions, samples):
  y = tf.math.sigmoid(predictions)
  return tf.sqrt(y * (1.0 - y)) * samples

def sample_mse_hessian(predictions, samples):
  #var = tf.math.reduce_std()
  return 0.00001 * samples

class CustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    wandb.log({'loss/test': logs["val_loss"], 'accuracy/test': logs["val_accuracy"]}, step=epoch)
    wandb.log({'loss/train': logs["loss"], 'accuracy/train': logs["accuracy"]}, step=epoch)

def make_optimizer(optimizer_name, hparams):
  #hp_lr = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128]
  #hp_momentum = [0.0, 0.2, 0.6, 0.8, 0.9, 0.95, 0.99]
  def sgd():
    return tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'])
  def sgdm():
    return tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=hparams['momentum'])
  def adam():
    return tf.keras.optimizers.Adam()
  def blockwise_trace():
    return KalmanTraceBlockwiseOptimizer(fading=1.0)
  def blockise_spectral():
    return KalmanSpectralBlockwiseOptimizer(fading=1.0, min_rank=hparams['min_rank'], max_rank=hparams['max_rank'])
  optimizer_map = {
    "sgd": sgd,
    "sgdm": sgdm,
    "adam": adam,
    "blockwise-trace": blockwise_trace,
    "blockwise-spectral": blockise_spectral
  }
  return optimizer_map[optimizer_name]()

def make_train_step(step_name):
  if (step_name == "default"):
    return DefaultTrainStep()
  if (step_name == "batch-fisher"):
    return FisherBatchTrainStep()
  if (step_name == "single-fisher"):
    return FisherSingleTrainStep()

def run_imdb(train_step_fn, optimizer, args):
  dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
  train_ds, test_ds = dataset['train'], dataset['test']

  train_ds = train_ds.shuffle(10000).batch(args['batch_size']).prefetch(1)
  test_ds = test_ds.batch(args['batch_size']).prefetch(1)
  
  VOCAB_SIZE=1000
  encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
  encoder.adapt(train_ds.map(lambda text, label: text))

  accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0)
  loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  
  model = RNNModel_IMDB(encoder, train_step_fn, sample_binary_crossentropy_hessian)
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy])
  model.fit(train_ds, epochs=args['epochs'], validation_data=test_ds, callbacks=[CustomCallback()])

def run_mnist_autoencoder(train_step_fn, optimizer, args):
  ## MNIST ##
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  print(x_train.shape)
  x_train = tf.reshape(x_train, [x_train.shape[0], -1])
  x_test = tf.reshape(x_test, [x_test.shape[0], -1])

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, x_train)).shuffle(10000).batch(args['batch_size']).prefetch(1)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(args['batch_size']).prefetch(1)

  accuracy = tf.keras.metrics.MeanSquaredError(name='accuracy')
  loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

  model = AutoEncodeMNIST(train_step_fn, sample_mse_hessian)
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy], run_eagerly=False)
  model.fit(train_ds, epochs=args['epochs'], validation_data=test_ds, callbacks=[CustomCallback()])

def run_cifar10(train_step_fn, optimizer, args):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(args['batch_size']).prefetch(tf.data.AUTOTUNE)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args['batch_size']).prefetch(tf.data.AUTOTUNE)

  accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

  model = VGGModel(train_step_fn, sample_crossentropy_hessian)
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy], run_eagerly=False)
  model.fit(train_ds, epochs=args['epochs'], validation_data=test_ds, callbacks=[CustomCallback()])

def run_benchmark(args):
  experiment_name = args['experiment_name']
  if experiment_name is None:
    experiment_name = wandb.util.generate_id()
  wandb.init(project="test-kalman", group=experiment_name,
    config={
      "engine": args['engine'],
      "optimizer": args['optimizer'],
      "batch_size": args['batch_size'],
      "learning_rate": args['learning_rate'] if 'learning_rate' in args else 0,
      "momentum": args['momentum'] if 'momentum' in args else 0,
      "min_rank": args['min_rank'] if 'min_rank' in args else 0,
      "max_rank": args['max_rank'] if 'max_rank' in args else 0,
    })

  optimizer = make_optimizer(args['optimizer'], args)
  train_step_fn = make_train_step(args['engine'])
  
  #run_mnist_autoencoder(train_step_fn, optimizer, args)
  run_cifar10(train_step_fn, optimizer, args)
  #run_imdb(train_step_fn, optimizer, args)

  wandb.finish()
  tf.keras.backend.clear_session()
  

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--engine', default='default')
  parser.add_argument('--optimizer', default='sgdm')
  parser.add_argument('--learning_rate', type=float, default=0.0005)
  parser.add_argument('--momentum', type=float, default=0.99)
  parser.add_argument('--min_rank', type=int, default=54)
  parser.add_argument('--max_rank', type=int, default=54+8)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--experiment_name')
  args = parser.parse_args()

  run_benchmark(vars(args))

if __name__ == "__main__":
  main()
