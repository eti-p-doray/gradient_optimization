import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

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

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    #self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(200, activation='relu')
    self.d2 = Dense(200, activation='relu')
    self.d3 = Dense(10)

  def call(self, x):
    #x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

# Create an instance of the model
N = 30
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

lr = 0.1
momentum=0.5
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

class MovingAverage(tf.keras.metrics.Metric):
  def __init__(self, name='moving_average', shape=(), decay=0.999, **kwargs):
    super(MovingAverage, self).__init__(name=name, **kwargs)
    self.decay = decay
    self.avg = self.add_weight(name='avg', shape=shape, initializer='zeros')

  def update_state(self, value):
    value = tf.cast(value, dtype=tf.float32)
    self.avg.assign(self.avg * self.decay + (1.0 - self.decay) * value)

  def result(self):
    return self.avg

#train_gradients_norm = tf.keras.metrics.Mean(name='train_gradients_norm')
#magnitude_change = tf.keras.metrics.Mean(name='magnitude_change')
#predict_gradients_norm = tf.keras.metrics.Mean(name='predict_gradients_norm')
gradient_norm = MovingAverage(decay=0.95)
magnitude_change = MovingAverage(decay=0.95)

baseline_avg = MovingAverage(shape=(199210), decay=0.999)

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
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  training_gradients = gradients

  gradient_vec = tf.concat([tf.reshape(gradient, [-1]) for gradient in training_gradients], axis=0)
  variables_vec = tf.concat([tf.reshape(trainable, [-1]) for trainable in model.trainable_variables], axis=0)
  baseline_avg(variables_vec)
  Or = 0.5 * lr * (1.0+momentum) * tf.tensordot(gradient_vec, gradient_vec, 1)
  Ol = -tf.tensordot(variables_vec, -gradient_vec, 1)

  gradient_norm(Or)
  magnitude_change(Ol)

  optimizer.apply_gradients(zip(training_gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
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
    print(
      f'Gradient Norm: {gradient_norm.result()}, '
      f'Magnitude Change: {magnitude_change.result()}, '
      f'Relation: {tf.abs(magnitude_change.result() / gradient_norm.result() - 1.0)}, '
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
