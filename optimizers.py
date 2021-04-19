import tensorflow as tf

def vec(weight):
  return tf.reshape(weight, [tf.shape(weight)[0], -1])

class DefaultTrainStep():
  def __init__(self):
    pass

  def __call__(self, model, data):
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = model(x, training=True)
      loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)
    gradients = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    model.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in model.metrics}

class FisherBatchTrainStep():
  def __init__(self):
    self.normal_initializer = tf.random_normal_initializer(stddev=1.0)

  @tf.function
  def __call__(self, model, data):
    x, y = data
    batch_size = tf.shape(x)[0]
    with tf.GradientTape() as tape:
      y_pred = model(x, training=True)
      loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = [g * tf.cast(batch_size, tf.float32) for g in gradients]
    
    sample = model.sample_observations(y_pred, self.normal_initializer(tf.shape(y_pred)))
    def sample_fim(x_and_sample):
      with tf.GradientTape() as tape:
        y_pred = model(tf.expand_dims(x_and_sample[0], 0), training=True)
      return tape.gradient(y_pred, model.trainable_weights, output_gradients=tf.expand_dims(x_and_sample[1], 0))
    fim_samples = tf.vectorized_map(sample_fim, (x, sample))
    fim_diagonal = model.fim_diagonal(x)

    with tf.init_scope():
      model.optimizer._create_slots(model.trainable_weight_initializers())

    model.optimizer.apply_information(zip(fim_samples, fim_diagonal, model.trainable_weights))
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    model.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in model.metrics}

class FisherSingleTrainStep():
  def __init__(self):
    self.normal_initializer = tf.random_normal_initializer(stddev=1.0)

  def __call__(self, model, data):
    x, y = data
    batch_size = tf.shape(x)[0]
    with tf.GradientTape() as tape:
      y_pred = model(x, training=True)
      loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = [g * tf.cast(batch_size, tf.float32) for g in gradients]

    sample = self.normal_initializer(tf.shape(y_pred))# * tf.sqrt(tf.cast(y_pred, tf.float32))
    sample = model.sample_observations(y_pred, sample)
    sample = tf.reduce_sum(sample, 0, keepdims=True)

    single_x = tf.expand_dims(x[0,], axis=0)
    with tf.GradientTape() as tape:
      observations = model(single_x, training=True)
    fim_samples = tape.gradient(observations, model.trainable_weights, output_gradients=sample)
    fim_samples = [tf.expand_dims(samples, 0) for samples in fim_samples]
    fim_diagonal = model.fim_diagonal(x)

    with tf.init_scope():
      model.optimizer._create_slots(model.trainable_weight_initializers())

    model.optimizer.apply_information(zip(fim_samples, fim_diagonal, model.trainable_weights))
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    model.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in model.metrics}

class BasicKalmanTraceOptimizer():
  def __init__(self, fading):
    self.fading = fading

  def create_slots(self, weight, initializer):
    var = tf.math.reduce_variance(weight)
    with tf.name_scope("precision-trace") as scope:
      print(weight.name, var, 1.0 / (0.001 + var), initializer)
      self.trace_norm = tf.Variable(initializer, name=weight.name)

  def apply_information(self, fisher_samples, information_scalar, weight):
    manifold = tf.reduce_sum(fisher_samples * fisher_samples)
    self.trace_norm.assign_add(manifold / tf.cast(tf.size(weight), tf.float32) + tf.reduce_sum(information_scalar))

  def apply_gradients(self, gradient, weight):
    weight.assign_add(-gradient / self.trace_norm)
    self.trace_norm.assign(self.trace_norm * self.fading)
    
  def summarize(self, step):
    wandb.log({self.trace_norm.name: self.trace_norm}, step=step)

class KalmanTraceBlockwiseOptimizer(tf.keras.optimizers.Optimizer):
  def __init__(self, fading):
    super(KalmanTraceBlockwiseOptimizer, self).__init__(name='kalman-blockwise-trace')
    self.fading = fading
    self.block_optimizers = None

  def _create_slots(self, var_list):
    def create_block_optimizer(weight, initializer):
      optimizer = BasicKalmanTraceOptimizer(self.fading)
      optimizer.create_slots(weight, initializer)
      return optimizer
    if (self.block_optimizers is None):
      self.block_optimizers = {}
      for weight, initializer in var_list:
        self.block_optimizers[weight.name] = create_block_optimizer(weight, initializer)

  def apply_information(self, information_and_weights):
    for fisher_samples, information_diagonal, weight in information_and_weights:
      optimizer = self.block_optimizers[weight.name]
      optimizer.apply_information(fisher_samples, information_diagonal, weight)

  def apply_gradients(self, gradients_and_weights):
    for gradient, weight in gradients_and_weights:
      optimizer = self.block_optimizers[weight.name]
      optimizer.apply_gradients(gradient, weight)

  def summarize(self, step):
    for optimizer in self.block_optimizers:
      optimizer.summarize(step)

  def get_config(self):
    return {"fading": self.fading}

class BasicKalmanSpectralOptimizer():
  def __init__(self, fading, min_rank, max_rank):
    self.fading = fading
    self.min_rank = min_rank
    self.max_rank = max_rank
    self.orthgonal_initializer = tf.keras.initializers.Orthogonal()

  def create_slots(self, weight, initializer):
    with tf.name_scope("precision-trace") as scope:
      self.trace_norm = tf.Variable(initializer, name=weight.name)
    with tf.name_scope("manifold") as scope:
      self.manifold = tf.Variable(tf.zeros([self.max_rank, self.max_rank]), name=weight.name)
    self.basis = tf.Variable(tf.zeros([self.max_rank, tf.size(weight)]))
    self.current_rank = tf.Variable(0, dtype=tf.int32)

  def augment(self, fisher_samples):
    n = tf.shape(fisher_samples)[0]
    B = tf.tensordot(self.basis[:-n,:], vec(fisher_samples), [[1], [1]])
    D = tf.tensordot(vec(fisher_samples), vec(fisher_samples), [[1], [1]])
    augmented_manifold = tf.concat([tf.concat([D, B], 0), tf.concat([tf.transpose(B), self.manifold[:-n,:-n]], 0)], 1)
    augmented_basis = tf.concat([vec(fisher_samples), self.basis[:-n,:]], 0)
    self.manifold.assign(augmented_manifold)
    self.basis.assign(augmented_basis)

  def compress(self, weight):
    rank = self.min_rank
    padding = self.max_rank - self.min_rank

    transform = self.orthgonal_initializer([rank,rank])
    with tf.device('/cpu:0'):
      s, v = tf.linalg.eigh(self.manifold)
      s = tf.math.maximum(s, 0)
    manifold_spill = tf.reduce_sum(s[:-rank]) / tf.cast(tf.size(weight), tf.float32)
    self.trace_norm.assign_add(manifold_spill)

    compressed_basis = tf.tensordot(
      tf.tensordot(v[:,-rank:], tf.transpose(transform), axes=1), 
      vec(self.basis), axes=[[0],[0]])
    compressed_manifold = tf.tensordot(compressed_basis, compressed_basis, [[1], [1]])

    compressed_basis = tf.pad(compressed_basis, [[0, padding], [0, 0]])
    compressed_manifold = tf.pad(compressed_manifold, [[0, padding], [0, padding]])
    self.manifold.assign(compressed_manifold)
    self.basis.assign(compressed_basis)

  @tf.function
  def apply_information(self, fisher_samples, information_diagonal, weight):
    if (self.current_rank + tf.shape(fisher_samples)[0] > self.max_rank):
      self.compress(weight)
      self.current_rank.assign(self.min_rank)

    self.current_rank.assign_add(tf.shape(fisher_samples)[0])
    self.trace_norm.assign_add(tf.reduce_sum(information_diagonal))
    self.augment(fisher_samples)

  def apply_gradients(self, gradient, weight):
    Xg = tf.tensordot(self.basis, tf.reshape(gradient, [-1]), [[1], [0]])
    with tf.device('/cpu:0'):
      Xg = tf.linalg.solve(self.manifold + self.trace_norm * tf.eye(self.manifold.shape[0]), tf.expand_dims(Xg, 1))
      Xg = tf.squeeze(Xg, 1)
    natural_gradient = tf.tensordot(self.basis, Xg, [[0], [0]])
    natural_gradient = tf.reshape(natural_gradient, gradient.shape)

    weight.assign_add((-gradient + natural_gradient) / self.trace_norm)
    
  def summarize(self, step):
    wandb.log({self.trace_norm.name: self.trace_norm}, step=step)

class KalmanSpectralBlockwiseOptimizer(tf.keras.optimizers.Optimizer):
  def __init__(self, fading, min_rank, max_rank):
    super(KalmanSpectralBlockwiseOptimizer, self).__init__(name='kalman-blockwise-trace')
    self.fading = fading
    self.min_rank = min_rank
    self.max_rank = max_rank
    self.block_optimizers = None

  def _create_slots(self, var_list):
    def create_block_optimizer(weight, initializer):
      optimizer = BasicKalmanSpectralOptimizer(self.fading, self.min_rank, self.max_rank)
      optimizer.create_slots(weight, initializer)
      return optimizer
    if (self.block_optimizers is None):
      self.block_optimizers = {}
      for weight, initializer in var_list:
        self.block_optimizers[weight.name] = create_block_optimizer(weight, initializer)

  def apply_information(self, information_and_weights):
    for fisher_samples, information_diagonal, weight in information_and_weights:
      optimizer = self.block_optimizers[weight.name]
      optimizer.apply_information(fisher_samples, information_diagonal, weight)

  def apply_gradients(self, gradients_and_weights):
    for gradient, weight in gradients_and_weights:
      optimizer = self.block_optimizers[weight.name]
      optimizer.apply_gradients(gradient, weight)

  def summarize(self, step):
    for optimizer in self.block_optimizers:
      optimizer.summarize(step)

  def get_config(self):
    return {
      "fading": self.fading,
      "min_rank": self.min_rank,
      "max_rank": self.max_rank
    }
