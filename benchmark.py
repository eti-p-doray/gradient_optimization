import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


class DefaultTrainEngine():
  def __init__(self, model, loss, accuracy, optimizer):
    self.optimizer = optimizer
    self.model = model
    self.loss = loss
    self.accuracy = accuracy

    self.loss_metric = tf.keras.metrics.Mean(name='loss')

  def reset_states():
    self.loss_metric.reset_states()
    self.accuracy.reset_states()

  @tf.function
  def train_step(self, x, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(x, training=True)
      loss = self.loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    self.accuracy(labels, predictions)
    self.loss_metric(loss)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

  @tf.function
  def train_step(self, x, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = self.model(x, training=False)
    loss = self.loss(labels, predictions)

    self.loss_metric(loss)
    self.accuracy(labels, predictions)
  

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

batch_size = 16
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

EPOCHS = 10

model = tf.keras.applications.ResNet50(input_shape=x_train.shape[1:], weights=None)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

engine = DefaultTrainEngine(model, loss_fn, train_accuracy, optimizer)

for epoch in range(EPOCHS):

  engine.reset_states()
  for x, labels in train_ds:
    engine.step(x, labels)
    print(
      f'Loss: {engine.loss_metric.result()}, '
      f'Accuracy: {train_accuracy.result() * 100}, '
    )

  engine.reset_states()
  for x, labels in test_ds:
    test_engine.step(x, labels)
  print(
    f'Loss: {engine.loss_metric.result()}, '
    f'Accuracy: {engine.result() * 100}, '
  )
