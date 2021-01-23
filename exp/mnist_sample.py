import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

ensemble_size = 4
batch_size = 4
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

normal_initializer = tf.random_normal_initializer()

class MNISTModel(Model):
  def __init__(self):
    super(MNISTModel, self).__init__()

    #self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d2 = Dense(50, activation='relu')
    self.d3 = Dense(10)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], 10])

  def get_config(self):
    return {}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def call(self, x):
    #x = self.conv1(x)
    x = self.flatten(x)
    #x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)

noise_initializer = tf.keras.initializers.VarianceScaling()

class EnsembleLayer(tf.keras.layers.Layer):
  def __init__(self, base_layer, ensemble_size):
    super(EnsembleLayer, self).__init__()
    self.base_layer = base_layer
    #self.base_layer.trainable = False
    self.noise_initializer = tf.keras.initializers.VarianceScaling()
    self.ensemble_size = ensemble_size

  def build(self, input_shape):
    self.base_layer.build(input_shape)
    self.ensemble_layers = [
      self.base_layer.from_config(self.base_layer.get_config()) for i in range(self.ensemble_size)
    ]
    for layer in self.ensemble_layers:
      layer.build(input_shape)
      for param, param_init in zip(layer.trainable_variables, self.base_layer.trainable_variables):
        param.assign(param_init + 0.1 * self.noise_initializer(param.shape))
      layer.trainable = False

    self.ensemble_params = [
      tf.Variable(tf.stack([layer.variables[i] for layer in self.ensemble_layers]), trainable=False) 
        for i, var in enumerate(self.base_layer.trainable_variables)]

  def call(self, input, training=None):
    if training:
      output = ([self.base_layer(input)] +
        [layer(tf.expand_dims(input[i], axis=0)) for i, layer in enumerate(model.ensemble_layers)])
      return output
    return self.base_layer(input)

model = EnsembleLayer(MNISTModel(), ensemble_size)
model.build(x_train.shape)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.SGD()

gradient_norm = tf.keras.metrics.Mean(name='gradient_norm')
gradient_var = tf.keras.metrics.Mean(name='gradient_var')
gradient_var2 = tf.keras.metrics.Mean(name='gradient_var2')
param_norm = tf.keras.metrics.Mean(name='param_norm')
param_var = tf.keras.metrics.Mean(name='param_var')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#@tf.function
def train_step(images, labels):
  with tf.GradientTape(persistent=True) as tape:
    predictions = model.call(images, training=True)
    #tiled_labels = tf.repeat(labels, repeats=[ensemble_size], axis=0)
    loss = loss_object(labels, tf.stack(predictions[1:], axis=0))
  #print([var.name for var in model.trainable_variables])
  #print([var.name for var in model.ensemble_params])
  gradients = tape.gradient(loss, predictions[1:])
  Hs = tape.jacobian(predictions[0], model.base_layer.trainable_variables)
  A_s = []
  for H, param in zip(Hs, model.ensemble_params):
    X = tf.reshape(param, [ensemble_size, -1])
    H = tf.reshape(tf.reduce_mean(H, axis=0), [predictions[0].shape[1], -1])
    X_ = X - tf.reduce_mean(X, axis=0, keepdims=True)
    #print(X_.shape, H.shape)
    b = tf.tensordot(H, X_, axes=[[1], [1]])
    #print(b.shape)
    A_s.append(b)
  A_ = tf.add_n(A_s)
  #print(A_.shape)
  #print(predictions[0].shape)

  train_loss(loss)
  train_accuracy(labels, predictions[0])
  
  #print(gradients[0].shape)
  Y = tf.transpose(tf.concat(predictions[1:], axis=0))
  Z = tf.transpose(-tf.concat(gradients, axis=0))
  #print(Y.shape, Z.shape)
  #A = tf.transpose(tf.linalg.diag_part(tiled_predictions))
  #print('Z', Z.shape)
  #print('Y', Y.shape)
  #print(Y, Z)

  Y_ = Y - tf.reduce_mean(Y, axis=1, keepdims=True)
  
  gradient_var(tf.reduce_sum(A_ * A_))
  gradient_norm(tf.reduce_sum(Z * Z))
  # HA is  ensemble_size

  s, u, v = tf.linalg.svd(Y_)
  X1 = tf.linalg.matmul(tf.linalg.diag(tf.divide(1.0, tf.math.square(s))), tf.transpose(u))
  X2 = tf.linalg.matmul(X1, Z)
  X3 = tf.linalg.matmul(u, X2)
  K = tf.linalg.matmul(tf.transpose(A_), X3)

  gradient_var2(tf.reduce_sum(K * K))
  for i, param in enumerate(model.ensemble_params):
    #print('param', param)
    X = tf.reshape(param, [ensemble_size, -1])
    A = X - tf.reduce_mean(X, axis=0, keepdims=True)
    X_ = tf.linalg.matmul(tf.transpose(K), A)
    A_ = X_ - tf.reduce_mean(X_, axis=0, keepdims=True)
    param_var(tf.reduce_sum(A * A) / X.shape[1])
    param_norm(tf.reduce_sum(X * X) / X.shape[1])
    #print('X, A', X, A)
    X = X + X_
    X_hat = tf.reduce_mean(X_, axis=0)
    
    #print('X, A', X, A)
    param.assign(tf.reshape(X, param.shape) + tf.stack([
          0.001 * noise_initializer(param.shape[1:]) 
        for i in range(ensemble_size)], axis=0))
    #print(param.name)

    for j, layer in enumerate(model.ensemble_layers):
      layer.variables[i].assign(param[j])
    model.base_layer.variables[i].assign(tf.reshape(X_hat, param.shape[1:]))
    

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
      f'Gradient Var: {gradient_var.result()}, '
      f'Gradient Var2: {gradient_var2.result()}, '
      f'Param Var: {param_var.result()}, '
      f'Gradient Norm: {gradient_norm.result()}, '
      f'Param Norm: {param_norm.result()}, '
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
