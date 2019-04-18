
import tensorflow as tf

y2 = tf.convert_to_tensor([2], dtype=tf.int64)
y_2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_2, labels=y2)

with tf.Session() as sess:
    print('c2: ' , sess.run(c2))