import tensorflow as tf
from datetime import datetime
import argparse
import math
import json
import wandb

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

class BaseEngine():
  def __init__(self, model, loss_fn, accuracy, optimizer):
    self.optimizer = optimizer
    self.model = model
    self.loss_fn = loss_fn
    self.accuracy = accuracy

    self.loss_metric = tf.keras.metrics.Mean(name='loss')

  def reset_states(self):
    self.loss_metric.reset_states()
    self.accuracy.reset_states()

  @tf.function
  def test_step(self, x, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = self.model(x, training=False)
    loss = self.loss_fn(labels, predictions)

    self.loss_metric(loss)
    self.accuracy(labels, predictions)

  def summarize(self, step):
    pass

class DefaultEngine(BaseEngine):
  def __init__(self, model, loss_fn, accuracy, optimizer):
    super(DefaultEngine, self).__init__(model, loss_fn, accuracy, optimizer)

  @tf.function
  def train_step(self, x, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(x, training=True)
      loss = self.loss_fn(labels, predictions)
    gradients = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

    self.accuracy(labels, predictions)
    self.loss_metric(loss)

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

class FisherBlockScalarOptimizer():
  def __init__(self, weights, fading):
    def initialize_information_norm(weight):
      fan_in, fan_out = compute_fans(weight.shape)
      return tf.Variable(tf.cast((fan_in + fan_out) / 2.0, tf.float32), name=weight.name)

    self.fading = fading

    with tf.name_scope("fisher-norm") as scope:
      self.block_norm = [initialize_information_norm(weight) for weight in weights]

  @tf.function
  def apply_update(self, gradients, weights, samples, size):
    for sample, norm, weight, gradient in zip(samples, self.block_norm, weights, gradients):
      embedding = tf.reduce_sum(sample * sample)
      
      norm.assign_add(embedding / tf.cast(tf.size(weight), tf.float32))
      weight.assign_add(-gradient * size / norm)
      norm.assign(norm * self.fading)

  def summarize(self, step):
    for norm in self.block_norm:
      wandb.log({"block-norm-" + norm.name: norm}, step=step)
      wandb.log({"block-lr-" + norm.name: 1.0 / norm}, step=step)

@tf.function
def reshape(weight, shape):
  shape = [weight_size if size is None else size for size, weight_size in zip(shape, tf.shape(weight))]
  return tf.reshape(weight, shape)

@tf.function
def vec(weight):
  return tf.reshape(weight, [tf.shape(weight)[0], -1])

@tf.function
def dot(A, B, axes):
  return tf.tensordot(A, B, axes)

class FisherBlockSpectralOptimizer():
  def __init__(self, weights, fading, rank):
    def initialize_information_norm(weight):
      fan_in, fan_out = compute_fans(weight.shape)
      return tf.Variable(tf.cast((fan_in + fan_out) / 2.0, tf.float32), name=weight.name)

    self.fading = fading
    self.rank = rank
    self.orthgonal_initializer = tf.keras.initializers.Orthogonal()

    with tf.name_scope("fisher-norm") as scope:
      self.block_norm = [initialize_information_norm(weight) for weight in weights]

    self.block_riemann_metric = [tf.Variable(tf.zeros([self.rank, self.rank])) for weight in weights]
    self.block_fisher_vectors = [tf.Variable(tf.zeros([self.rank] + weight.shape)) for weight in weights]

  @tf.function
  def augment(self, riemann_metric, fisher_vectors, samples):
    #print(riemann_metric.shape, fisher_vectors.shape, samples.shape)
    #print(tf.tensordot(vec(fisher_vectors), vec(fisher_vectors), [[1], [1]]))
    #print("riemann_metric", riemann_metric)
    B = tf.tensordot(vec(fisher_vectors), vec(samples), [[1], [1]])
    D = tf.tensordot(vec(samples), vec(samples), [[1], [1]])
    #print('BD', B, D)
    #print('riemann_metric', riemann_metric)
    augmented_riemann_metric = tf.concat([tf.concat([riemann_metric, tf.transpose(B)], 0), tf.concat([B, D], 0)], 1)
    #print('augmented_riemann_metric', augmented_riemann_metric)
    augmented_fisher_vector = tf.concat([fisher_vectors, samples], 0)
    #print(tf.tensordot(vec(augmented_fisher_vector), vec(augmented_fisher_vector), [[1], [1]]))
    #print("augmented_riemann_metric", augmented_riemann_metric)
    return augmented_riemann_metric, augmented_fisher_vector

  @tf.function
  def compress(self, riemann_metric, fisher_vectors, weight, rank):
    transform = self.orthgonal_initializer([rank,rank])

    s, u, v = tf.linalg.svd(riemann_metric)
    #print(s, v)
    metric_spill = tf.reduce_sum(s[rank:]) / tf.cast(tf.size(weight), tf.float32)
    #print(s)

    #a, b, c = tf.linalg.svd(vec(fisher_vectors))
    #print(a, c)

    compressed_fisher_vectors = tf.tensordot(tf.tensordot(v[:,0:rank], tf.transpose(transform), axes=1), vec(fisher_vectors), axes=[[0],[0]])
    compressed_fisher_vectors = tf.reshape(compressed_fisher_vectors, [rank] + weight.shape)
    
    #print(tf.expand_dims(s[0:rank], axis=0).shape)
    #print('riemann_metric', riemann_metric)
    compressed_riemann_metric = tf.tensordot(transform * tf.expand_dims(s[0:rank], axis=0), tf.transpose(transform), axes=1)
    #print(tf.tensordot(vec(compressed_fisher_vectors), vec(compressed_fisher_vectors), [[1], [1]]))
    #print("compressed_riemann_metric", compressed_riemann_metric)
    #print('compressed_riemann_metric', compressed_riemann_metric)
    return compressed_riemann_metric, compressed_fisher_vectors, metric_spill

  @tf.function
  def apply_update(self, gradients, weights, block_samples, batch_size):
    for samples, norm, fisher_vectors, riemann_metric, weight, gradient in zip(block_samples, self.block_norm, self.block_fisher_vectors, self.block_riemann_metric, weights, gradients):
      augmented_riemann_metric, augmented_fisher_vector = self.augment(riemann_metric, fisher_vectors, samples)
      #print(gradient.shape)
      
      #print(Xg.shape)

      compressed_riemann_metric, compressed_fisher_vector, metric_leak = self.compress(augmented_riemann_metric, augmented_fisher_vector, weight, self.rank)
      #print(compressed_riemann_metric.shape, compressed_fisher_vector.shape)
      riemann_metric.assign(compressed_riemann_metric)
      fisher_vectors.assign(compressed_fisher_vector)
      #print(norm)
      norm.assign_add(metric_leak)
      #norm.assign(norm * self.fading)

      inverse_metric = tf.linalg.inv(compressed_riemann_metric + norm * tf.eye(self.rank))
      #print(fisher_vectors.shape, gradient.shape)
      Xg = tf.tensordot(vec(compressed_fisher_vector), tf.reshape(gradient, [-1]), [[1], [0]])
      #print(Xg.shape)
      Xg = tf.tensordot(inverse_metric, Xg, 1)
      #print(Xg.shape)
      natural_gradient = tf.tensordot(compressed_fisher_vector, Xg, [[0], [0]])

      weight.assign_add((-gradient + natural_gradient) * batch_size / norm)

  def summarize(self, step):
    for norm in self.block_norm:
      wandb.log({"block-norm-" + norm.name: norm}, step=step)
      wandb.log({"block-lr-" + norm.name: 1.0 / norm}, step=step)

loss_fn2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

class BatchFisherEngine(BaseEngine):
  def __init__(self, model, loss_fn, hessian_fn, accuracy, optimizer):
    super(BatchFisherEngine, self).__init__(model, loss_fn, accuracy, optimizer)
    self.hessian_fn = hessian_fn
    self.orthgonal_initializer = tf.keras.initializers.Orthogonal()
    self.normal_initializer = tf.random_normal_initializer(stddev=1.0)

  @tf.function
  def train_step(self, x, labels):
    with tf.GradientTape(persistent=True) as tape:
      predictions = self.model(x, training=True)
      single_prediction = tf.expand_dims(predictions[0,:], 0)
      loss = self.loss_fn(labels, predictions)

    #samples = self.orthgonal_initializer(predictions.shape)
    samples = self.normal_initializer(single_prediction.shape) * tf.sqrt(tf.cast(x.shape[0], tf.float32))
    hessian = self.hessian_fn(single_prediction)
    samples = tf.linalg.matmul(hessian, tf.expand_dims(samples, 2))
    samples = tf.squeeze(samples, 2)
    information_samples = tape.gradient(single_prediction, self.model.trainable_weights, output_gradients=samples)
    gradients = tape.gradient(loss, self.model.trainable_weights)

    self.optimizer.apply_update(gradients, self.model.trainable_weights, information_samples, x.shape[0])

    self.accuracy(labels, predictions)
    self.loss_metric(loss)

  def summarize(self, step):
    self.optimizer.summarize(step)

class FisherEngine(BaseEngine):
  def __init__(self, model, loss_fn, hessian_fn, accuracy, optimizer):
    super(FisherEngine, self).__init__(model, loss_fn, accuracy, optimizer)
    self.hessian_fn = hessian_fn
    self.orthgonal_initializer = tf.keras.initializers.Orthogonal()
    self.normal_initializer = tf.random_normal_initializer(stddev=1.0)

  @tf.function
  def train_step(self, x, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(x, training=True)
      loss = self.loss_fn(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_weights)

    def backward_acc(sample_and_x):
      with tf.GradientTape() as tape:
        predictions = self.model(tf.expand_dims(sample_and_x[1], axis=0), training=True)
      sample = tf.expand_dims(sample_and_x[0], 0)
      return tape.gradient(predictions, self.model.trainable_weights, output_gradients=sample)

    #samples = self.orthgonal_initializer(predictions.shape) * x.shape[0]
    samples = self.normal_initializer(predictions.shape)
    hessian = self.hessian_fn(predictions)
    samples = tf.linalg.matmul(hessian, tf.expand_dims(samples, 2))
    samples = tf.squeeze(samples, 2)
    information_samples = tf.vectorized_map(backward_acc, [samples, x])

    self.optimizer.apply_update(gradients, self.model.trainable_weights, information_samples, x.shape[0])

    self.accuracy(labels, predictions)
    self.loss_metric(loss)

  def summarize(self, step):
    self.optimizer.summarize(step)

class CNNModel(tf.keras.Model):
  def __init__(self):
    super(CNNModel, self).__init__()

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

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class VGGModel(tf.keras.Model):
  def __init__(self):
    super(VGGModel, self).__init__()

    self.conv1_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch1_1 = tf.keras.layers.BatchNormalization()
    self.conv1_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch1_2 = tf.keras.layers.BatchNormalization()
    self.pool1 = tf.keras.layers.MaxPool2D()
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.conv2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch2_1 = tf.keras.layers.BatchNormalization()
    self.conv2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch2_2 = tf.keras.layers.BatchNormalization()
    self.pool2 = tf.keras.layers.MaxPool2D()
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    self.conv3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch3_1 = tf.keras.layers.BatchNormalization()
    self.conv3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform')
    #self.batch3_2 = tf.keras.layers.BatchNormalization()
    self.pool3 = tf.keras.layers.MaxPool2D()
    self.dropout3 = tf.keras.layers.Dropout(0.2)
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.dropout4 = tf.keras.layers.Dropout(0.2)
    self.d2 = tf.keras.layers.Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  def call(self, x):
    x = self.conv1_1(x)
    #x = self.batch1_1(x)
    x = self.conv1_2(x)
    #x = self.batch1_2(x)
    x = self.pool1(x)
    x = self.dropout1(x)

    x = self.conv2_1(x)
    #x = self.batch2_1(x)
    x = self.conv2_2(x)
    #x = self.batch2_2(x)
    x = self.pool2(x)
    x = self.dropout2(x)

    x = self.conv3_1(x)
    #x = self.batch3_1(x)
    x = self.conv3_2(x)
    #x = self.batch3_2(x)
    x = self.pool3(x)
    x = self.dropout3(x)

    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout4(x)
    return self.d2(x)

@tf.function
def crossentropy_hessian_fn(predictions):
  y = tf.nn.softmax(predictions)
  z = tf.sqrt(y)
  return (tf.linalg.diag(z) - tf.linalg.matmul(tf.expand_dims(y, 1), tf.expand_dims(z, 1), transpose_a=True))

def make_optimizer(model, optimizer_name, hparams):
  #hp_lr = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128]
  #hp_momentum = [0.0, 0.2, 0.6, 0.8, 0.9, 0.95, 0.99]
  def sgd():
    return tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'])
  def sgdm():
    return tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=hparams['momentum'])
  def adam():
    return tf.keras.optimizers.Adam()
  def block_scalar():
    return FisherBlockScalarOptimizer(model.trainable_weights, fading=1.0)
  def block_low_rank():
    return FisherBlockSpectralOptimizer(model.trainable_weights, fading=1.0, rank=32)
  optimizers = {
    "sgd": sgd,
    "sgdm": sgdm,
    "adam": adam,
    "block-scalar": block_scalar,
    "block-spectral": block_low_rank
  }
  return optimizers[optimizer_name]()

def make_engine(engine_name, model, loss_fn, hessian_fn, accuracy, optimizer):
  if (engine_name == "default"):
    return DefaultEngine(model, loss_fn, accuracy, optimizer)
  elif (engine_name == "batch-fisher"):
    return BatchFisherEngine(model, loss_fn, hessian_fn, accuracy, optimizer)
  elif (engine_name == "fisher"):
    return FisherEngine(model, loss_fn, hessian_fn, accuracy, optimizer)

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--engine', default='default')
  parser.add_argument('--optimizer', default='sgdm')
  parser.add_argument('--learning_rate', type=float, default=0.0001)
  parser.add_argument('--momentum', type=float, default=0.99)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--experiment_name')
  args = parser.parse_args()

  experiment_name = args.experiment_name
  if experiment_name is None:
    experiment_name = wandb.util.generate_id()
  wandb.init(project="kalman-fisher", group=experiment_name,
    config={
      "engine": args.engine,
      "optimizer": args.optimizer,
      "batch_size": args.batch_size,
      "learning_rate": args.learning_rate,
      "momentum": args.momentum,
    })

  hparams = {
    "learning_rate": args.learning_rate,
    "momentum": args.momentum
  }

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Add a channels dimension
  #x_train = x_train[..., tf.newaxis].astype("float32")
  #x_test = x_test[..., tf.newaxis].astype("float32")

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(args.batch_size).prefetch(1)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size).prefetch(1)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  hessian_fn = crossentropy_hessian_fn
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

  #model = CNNModel()
  model = VGGModel()
  #model = tf.keras.applications.ResNet50(weights=None, classes=10, input_shape=x_train.shape[1:])
  model.build(x_train.shape)

  optimizer = make_optimizer(model, args.optimizer, hparams)
  engine = make_engine(args.engine, model, loss_fn, hessian_fn, accuracy, optimizer)

  for epoch in range(args.epochs):
    engine.reset_states()
    print(
      f'Step: {epoch}'
    )
    for x, labels in train_ds:
      engine.train_step(x, labels)
    print(
      'Train '
      f'Loss: {engine.loss_metric.result()}, '
      f'Accuracy: {accuracy.result() * 100}, '
    )
    wandb.log({'accuracy/train': engine.accuracy.result()}, step=epoch)
    wandb.log({'loss/train': engine.loss_metric.result()}, step=epoch)
    engine.summarize(epoch)

    engine.reset_states()
    for x, labels in test_ds:
      engine.test_step(x, labels)
    print(
      'Test '
      f'Loss: {engine.loss_metric.result()}, '
      f'Accuracy: {accuracy.result() * 100}, '
    )
    wandb.log({'accuracy/test': engine.accuracy.result()}, step=epoch)
    wandb.log({'loss/test': engine.loss_metric.result()}, step=epoch)

  wandb.finish()


if __name__ == "__main__":
  main()
