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

ensemble_size = 32
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

    print([var.shape for var in self.base_layer.trainable_variables])
    self.ensemble_information = [tf.Variable(0.5*tf.stack([
          self.noise_initializer(param.shape)
        for i in range(ensemble_size)] , axis=0), trainable=False)
      for param in self.base_layer.trainable_variables]
    print([var.shape for var in self.ensemble_information])
    for A in self.ensemble_information:
      A.assign(A - tf.reduce_mean(A, axis=1, keepdims=True))

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

@tf.function
def train_step(images, labels):

  # Forward propagation
  def foward_acc(tangents):
    with tf.autodiff.ForwardAccumulator(
        model.trainable_variables,
        tangents) as acc:
      predictions = model(images, training=True)
    return acc.jvp(predictions)

  def backward_acc(tangents):
    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
    return tape.gradients(predictions, model.trainable_variables, output_gradients=tangents)

  # Backward propagation
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, [predictions] + model.base_layer.trainable_variables)
  train_loss(loss)
  train_accuracy(labels, predictions)
  
  #print(tf.transpose(noise))

  grad_norm = tf.reduce_sum([tf.reduce_sum(grad*grad) for grad in gradients[1:]])
  print(grad_norm)
  noise = 10.0*noise_initializer([ensemble_size, 1]) / grad_norm
  for A, grad in zip(model.ensemble_information, gradients[1:]):
    # print(tf.tensordot(noise, tf.reshape(grad, [1, -1]), axes=1).shape)
    A.assign_add(tf.reshape(tf.tensordot(noise, tf.reshape(grad, [1, -1]), axes=1), A.shape))
    A.assign(A - tf.reduce_mean(A, axis=1, keepdims=True))

  # S = Hx' projected estimate covariance ensemble
  S = tf.vectorized_map(foward_acc, model.ensemble_information)
  S = tf.transpose(tf.reshape(S, [ensemble_size, -1]))
  S = S - tf.reduce_mean(S, axis=1, keepdims=True)
  # E = S + observation covariance ensemble
  #E = noise_initializer(S.shape)
  #E = E - tf.reduce_mean(E, axis=1, keepdims=True)
  #print(tf.square(tf.linalg.svd(E, compute_uv=False)))
  #print(tf.square(tf.linalg.svd(S, compute_uv=False)))
  #train_mean(tf.norm(tf.reduce_mean(E, axis=1)))
  # E -> U * Sigma * V.t
  #Sigma, Upsilon, _ = tf.linalg.svd(E + S)
  print(tf.square(tf.linalg.svd(S, compute_uv=False)))
  Sigma, Upsilon, _ = tf.linalg.svd(tf.eye(S.shape[0])*(ensemble_size-1) + tf.tensordot(S, tf.transpose(S), axes=1))
  # Gamma = Upsilon * 1/Sigma
  #Gamma = tf.tensordot(Upsilon, tf.linalg.tensor_diag(1.0/Sigma), axes=1)
  Gamma = tf.tensordot(Upsilon, tf.linalg.tensor_diag(tf.sqrt(1.0/Sigma)), axes=1)
  # C = Gamma * Gamma.t Innovation Covariance
  # Phi = S.t * Gamma
  Phi = tf.tensordot(tf.transpose(S), Gamma, axes=1)
  # Phi-> Zeta * Lambda * Zeta.t
  Lambda, Zeta, _ = tf.linalg.svd(Phi, full_matrices=True)
  # Mu = Zeta * sqrt(1 - Lambda) * Orthogonal
  #print()
  print(Lambda)
  Mu = tf.tensordot(
    tf.tensordot(Zeta, tf.linalg.tensor_diag(tf.sqrt(tf.math.maximum(0.0, 1.0 - tf.square(Lambda)))), axes=1),
    tf.transpose(orthogonal(Zeta.shape)), 
    axes=1)

  # Delta = S.t * Gamma * Gamma.t * gradient
  Delta = tf.tensordot(
    tf.transpose(S), 
    tf.tensordot(
      Gamma, 
      tf.tensordot(tf.transpose(Gamma), -tf.reshape(gradients[0], [-1]), axes=1), 
      axes=1),
    axes=1)/0.2

  #print(tf.transpose(noise), Delta, tf.reduce_sum(noise * Delta))

  #gradients = tape.gradient(loss, model.trainable_variables)

  for A, param in zip(model.ensemble_information, model.base_layer.trainable_variables):
    param.assign_add(tf.reshape(tf.tensordot(tf.transpose(Delta), tf.reshape(A, [ensemble_size,-1]), axes=1), param.shape))
    A.assign(tf.reshape(tf.tensordot(tf.transpose(Mu), tf.reshape(A, [ensemble_size, -1]), axes=1), A.shape))
    #print(tf.norm(tf.reduce_mean(A, axis=1)))
    #A.assign(A - tf.reduce_mean(A, axis=1, keepdims=True))

  #rate.assign(rate / (1.0-decay+rate))
  #optimizer.apply_gradients(zip(gradients[1:], model.base_layer.trainable_variables))

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

