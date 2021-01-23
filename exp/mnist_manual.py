import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

normal_initializer = tf.random_normal_initializer()

class MyModel():
  def __init__(self):
    #self.conv1 = Conv2D(32, 3, activation='relu')
    self.weight_initializer = tf.keras.initializers.GlorotUniform()
    self.bias_initializer = tf.zeros_initializer()

    self.flatten = Flatten()
    #self.d1_a = tf.keras.layers.ReLU()
    #self.d2_a = tf.keras.layers.ReLU()

  def build(self, input_shape):
    self.d1_w = tf.Variable(self.weight_initializer(shape=[np.prod(input_shape[1:], dtype=int), 200]))
    self.d1_b = tf.Variable(self.bias_initializer(shape=[200]))

    self.d2_w = tf.Variable(self.weight_initializer(shape=[200, 200]))
    self.d2_b = tf.Variable(self.bias_initializer(shape=[200]))

    self.d3_w = tf.Variable(self.weight_initializer(shape=[200, 10]))
    self.d3_b = tf.Variable(self.bias_initializer(shape=[10]))

  def call(self, x, training = False):
    #x = self.conv1(x)
    x = self.flatten(x)
    x = tf.nn.relu( tf.tensordot(x, self.d1_w, [[1], [0]]) + self.d1_b )
    x = tf.nn.relu( tf.tensordot(x, self.d2_w, [[1], [0]]) + self.d2_b )
    x = tf.tensordot(x, self.d3_w, [[1], [0]]) + self.d3_b
    return x

  def trainable_variables(self):
    return [self.d1_w, self.d1_b, self.d2_w, self.d2_b, self.d3_w, self.d3_b]

# Create an instance of the model
N = 30
model = MyModel()
model.build(x_train.shape)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

lr = 0.1
momentum=0.5
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  print(images, labels)
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model.call(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables())
  training_gradients = gradients

  optimizer.apply_gradients(zip(training_gradients, model.trainable_variables()))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model.call(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 200

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

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
