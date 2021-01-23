import tensorflow as tf

a = tf.random.normal([128, 8])
print(tf.reduce_mean(a, axis=1, keepdims=True).shape)
a = a - tf.reduce_mean(a, axis=1, keepdims=True)

S = tf.tensordot(tf.transpose(a), a, axes=1)
P = tf.tensordot(a, tf.transpose(a), axes=1)
s1, u1, v1 = tf.linalg.svd(S)
print(v1.shape)
print(s1.shape)
print(s1[0:-1].shape)

#s2, u2, v2 = tf.linalg.svd(P)

#print('u_', v1.shape, tf.expand_dims(1.0/s1, axis=0).shape, (v1 * tf.expand_dims(1.0/tf.sqrt(s1), axis=0)).shape)
u_ = tf.tensordot(a, (v1 * tf.expand_dims(1.0/tf.sqrt(s1), axis=0)), axes=1)
#u_2 = tf.tensordot(tf.tensordot(a, v1, axes=1), tf.expand_dims(1.0/s1, axis=0), axes=[[1], [1]])
print(tf.tensordot(tf.transpose(u_), u_, axes=1))
#print(tf.tensordot(tf.transpose(u_2), u_2, axes=1))

#print(tf.tensordot(u1, tf.transpose(v1), axes=1))

#s2, u2, v2 = tf.linalg.svd(r)

#print(tf.reduce_sum(tf.square(u1 - tf.transpose(v1))))
#print(tf.reduce_sum(tf.square(v1 - v2)))
#print(s1)
#print(s2)



