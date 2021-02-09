import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib
import matplotlib.cm
import numpy as np
from datetime import datetime

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

ensemble_size = 64
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

normal_initializer = tf.random_normal_initializer()

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = tf.cast(tf.round(value * 255), tf.int32)

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'cividis')
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value

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


class MNISTModel(Model):
  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def __init__(self):
    super(MNISTModel, self).__init__()

    self.flatten = Flatten()
    self.conv1 = Conv2D(32, 5, activation='relu')
    self.pool1 = MaxPool2D()
    self.conv2 = Conv2D(64, 5, activation='relu')
    self.pool2 = MaxPool2D()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

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

model = MNISTModel()
model.build(x_train.shape)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#gauss_newton = tf.Variable(tf.zeros([10]), name='gauss_newton', trainable=False)
#gradient_samples = tf.Variable(tf.zeros([10]), name='gradient_samples', trainable=False)

@tf.function
def sample_gauss_newton(images):
  samples = orthgonal_initializer([batch_size, 10])
  def backward_acc(sample_and_image):
    with tf.GradientTape() as tape:
      predictions = model(tf.expand_dims(sample_and_image[1], axis=0), training=True)
    hessian = tf.math.sigmoid(predictions) * (1.0-tf.math.sigmoid(predictions))
    sample = tf.sqrt(hessian) * sample_and_image[0]
    return tape.gradient(predictions, model.trainable_variables, output_gradients=sample)

  grad = tf.vectorized_map(backward_acc, [samples, images])

  """with tf.GradientTape() as tape:
    predictions = model(tf.expand_dims(images[1], axis=0), training=True)
  hessian = tf.math.sigmoid(predictions) * (1.0-tf.math.sigmoid(predictions))
  sample = tf.sqrt(hessian) * samples
  grad = tape.gradient(predictions, model.trainable_variables, output_gradients=sample)"""
  return grad

class EnsembleBlockOptimizer:
  def __init__(self):
    pass

  def append_information(images):
    samples_list = sample_gauss_newton(images, self.hessian_fn)
    for embedding, ensemble, samples in zip(self.embedding_list, self.ensemble_list, samples_list):
      samples_embedding = tf.tensordot(samples, tf.transpose(samples), axes=1)
      cross_embedding = tf.tensordot(samples, tf.transpose(ensemble), axes=1)
      #embedding.assign(...)
      ensemble.assign(tf.concat([ensemble, samples], axis=0))


  def compress_information():
    pass

def initialize_information(shape):
  fan_in, fan_out = compute_fans(shape)
  return tf.Variable((fan_in + fan_out) / 2.0)
  
block_information = [initialize_information(var.shape) for var in model.trainable_variables]
print(block_information)
@tf.function
def train_step(images, labels):
  # Backward propagation
  #print(images)
  with tf.GradientTape(persistent=True) as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

  samples = orthgonal_initializer(predictions.shape)
  hessian = tf.math.sigmoid(predictions) * (1.0-tf.math.sigmoid(predictions))
  block_gradient_samples = tape.gradient(predictions, model.trainable_weights, output_gradients=tf.sqrt(hessian) * samples)

  gradients = tape.gradient(loss, model.trainable_weights)
  train_loss(loss)
  train_accuracy(labels, predictions)

  #block_gradient_samples = sample_gauss_newton(images)
  for gn, information, var, grad in zip(block_gradient_samples, block_information, model.trainable_weights, gradients):
    gn = tf.reshape(gn, [1,-1])
    embedding = tf.tensordot(gn, tf.transpose(gn), axes=1)
    information.assign_add(tf.linalg.trace(embedding)/tf.cast(tf.size(var), tf.float32))
    var.assign_add(-grad * images.shape[0] / information)
    #fan_in, fan_out = compute_fans(var.shape)
    #information.assign(information / (1.0 + information * 0.0001 / (fan_in+fan_out)))
  print(block_information)

  #optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

orthgonal_initializer = tf.keras.initializers.Orthogonal()

@tf.function
def analyse_step(images):
  samples = orthgonal_initializer([batch_size, 10])
  def backward_acc(sample_and_image):
    print(sample_and_image)
    with tf.GradientTape() as tape:
      predictions = model(tf.expand_dims(sample_and_image[1], axis=0), training=True)
    print(predictions)
    hessian = tf.math.sigmoid(predictions) * (1.0-tf.math.sigmoid(predictions))
    sample = tf.sqrt(hessian) * sample_and_image[0]
    return tape.gradient(predictions, model.trainable_variables, output_gradients=sample)

  grad = tf.vectorized_map(backward_acc, [samples, images])
  return grad

EPOCHS = 10
np.set_printoptions(threshold=5000)

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)
    print(
      f'Loss: {train_loss.result()}, '
      f'Accuracy: {train_accuracy.result() * 100}, '
    )

  print(block_information)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  """block_gradient_samples = [analyse_step(test_images) for test_images, test_labels in test_ds]
  block_gradient_samples = [tf.concat([tf.reshape(sample[i], [sample[i].shape[0],-1]) for sample in block_gradient_samples], axis=0, name=var.name) for i, var in enumerate(model.trainable_variables)]
  print([[grad.shape for grad in block_gradient_samples]])
  
  gradient_samples = tf.concat(block_gradient_samples, axis=1)

  gauss_newton = tf.tensordot(tf.transpose(gradient_samples), gradient_samples, axes=1)
  print(gauss_newton.shape)
  gradient_samples_mean = tf.reduce_mean(gradient_samples, axis=0, keepdims=True)
  print(tf.reduce_sum(gradient_samples_mean))
  gauss_newton_mean = tf.tensordot(tf.transpose(gradient_samples_mean), gradient_samples_mean, axes=1)
  gradient_samples_sum = tf.reduce_sum(tf.math.square(gradient_samples), axis=0, keepdims=True)
  gauss_newton_correlation = tf.math.abs(gauss_newton) / tf.math.sqrt(tf.tensordot(tf.transpose(gradient_samples_sum), gradient_samples_sum, axes=1))

  # Sets up a timestamped log directory.
  logdir = "train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  # Creates a file writer for the log directory.
  file_writer = tf.summary.create_file_writer(logdir)

  # Using the file writer, log the reshaped image.
  with file_writer.as_default():
    for var, block in zip(model.trainable_variables, block_gradient_samples):
      block_gauss_newton = tf.tensordot(tf.transpose(block), block, axes=1)
      with tf.device('/CPU:0'):
        singular_values = tf.linalg.svd(block_gauss_newton, compute_uv=False)
      tf.summary.histogram("Block singular values " + var.name, tf.math.log(singular_values+1), step=epoch)
      tf.summary.image("Block Gauss Newton " + var.name, tf.expand_dims(colorize(block_gauss_newton), axis=0), step=epoch)

    with tf.device('/CPU:0'):
      singular_values = tf.linalg.svd(gauss_newton, compute_uv=False)
    tf.summary.histogram("Singular values", tf.math.log(singular_values+1), step=epoch)
    tf.summary.image("Gauss Newton covariance", tf.expand_dims(colorize(gauss_newton), axis=0), step=epoch)
    tf.summary.image("Gauss Newton correlation", tf.expand_dims(colorize(gauss_newton_correlation), axis=0), step=epoch)
    tf.summary.image("Gauss Newton mean", tf.expand_dims(colorize(gauss_newton_mean), axis=0), step=epoch)"""


  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )


