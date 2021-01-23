import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

ensemble_size = 8
batch_size = 8
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

normal_initializer = tf.random_normal_initializer()


class MNISTModel(Model):
  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def __init__(self):
    super(MNISTModel, self).__init__()

    self.flatten = Flatten()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.conv2 = Conv2D(16, 3, activation='relu')
    self.conv3 = Conv2D(8, 3, activation='relu')
    self.d1 = Dense(200, activation='relu')
    self.d2 = Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class EnsembleLayer(tf.keras.layers.Layer):
  def __init__(self, base_layer, ensemble_size):
    super(EnsembleLayer, self).__init__()
    self.base_layer = base_layer
    self.noise_initializer = tf.keras.initializers.VarianceScaling()
    self.ensemble_size = ensemble_size

  def build(self, input_shape):
    self.base_layer.build(input_shape)

    #print([var.shape for var in self.base_layer.trainable_variables])
    self.ensemble_information = [tf.Variable(0.5*tf.stack([
          self.noise_initializer(param.shape)
        for i in range(ensemble_size)] , axis=0), trainable=False)
      for param in self.base_layer.trainable_variables]
    #print([var.shape for var in self.ensemble_information])
    for A in self.ensemble_information:
      A.assign(A - tf.reduce_mean(A, axis=1, keepdims=True))

    def inner_product(A):
      A = tf.reshape(A, [A.shape[0], -1])
      return tf.tensordot(A, tf.transpose(A), axes=1)

    self.ensemble_product = tf.Variable(tf.add_n([
      inner_product(A) for A in self.ensemble_information
    ]), trainable=False)
    #Sigma, Upsilon, _ = tf.linalg.svd(Gamma)
    #print(Sigma)

    #s, u, v = tf.linalg.svd(ensemble_information)
    #self.ensemble_information = tf.Variable(tf.tensordot(u, tf.linalg.diag(s), axes=1), trainable=False)

  def call(self, input, training=None):
    return self.base_layer(input, training=training)


model = EnsembleLayer(MNISTModel(), ensemble_size)
model.build(x_train.shape)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_mean = tf.keras.metrics.Mean(name='train_mean')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

noise_initializer = tf.random_normal_initializer(stddev=1)
orthogonal = tf.keras.initializers.Orthogonal()

decay = 0.1
rate = tf.Variable(0.5, trainable=False)

#@tf.function
def train_step(images, labels):
  # Backward propagation
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.base_layer.trainable_variables)
  train_loss(loss)
  train_accuracy(labels, predictions)

  Phi = tf.add_n([
    tf.reduce_sum(gradient * gradient) for gradient in gradients])
  probability = tf.math.exp(-loss)
  print(Phi, probability)

  #for param, gradient in zip(model.base_layer.trainable_variables, gradients):
  #  param.assign_add(-0.005*gradient / (Phi*probability))

  optimizer.apply_gradients(zip(gradients, model.base_layer.trainable_variables))

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

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

