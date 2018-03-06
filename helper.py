import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# help(tf.nn.dropout())


a = [1,1,1,1,0,0,1,1,0]

label_batch = tf.one_hot(a, depth=2)
label_batch = tf.reshape(label_batch, [9, 2])

with tf.Session() as sess:
   test = sess.run(label_batch)
   print(test)