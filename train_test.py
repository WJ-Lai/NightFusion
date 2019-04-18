import tensorflow as tf
import numpy as np
import os
from time import time

from model.factory import model_factory
from dataset.hazy_person import provider
import config
from utils.logging import logger

batch_size = 5

def main(_):

    inputs = tf.placeholder(tf.float32,
                            shape=(None, config.img_size[1], config.img_size[0], 3))
    label_gt = tf.placeholder(tf.int32,
                              shape=(None, config.grid_cell_size[0], config.grid_cell_size[1], 1))
    box_gt = tf.placeholder(tf.float32,
                              shape=(None, config.grid_cell_size[0], config.grid_cell_size[1], (9*2)))

    global_step = tf.Variable(0, trainable=False, name='global_step')

    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=42,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    output = tf.layers.max_pooling2d(inputs=conv, pool_size=[25, 25], strides=25)
    s = output.shape
    det = tf.slice(output, [0, 0, 0, 0], [-1, s[1], s[2], 18])
    det_loss = tf.reduce_mean(tf.square(tf.subtract(det, box_gt)))

    clf = tf.slice(output, [0, 0, 0, 18], [-1, s[1], s[2], 24])
    clf_pre = tf.reshape(clf, shape=[-1, 24])
    clf_gt = tf.reshape(label_gt, shape=[-1])
    clf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_pre, labels=clf_gt))

    loss = 5*det_loss+0.5*clf_loss

    pd = provider(batch_size=batch_size, for_what='train', whether_aug=True)
    imgs, labels, control_points = pd.load_batch()
    imgs = np.array(imgs)
    labels = np.array(labels)
    control_points = np.array(control_points)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(5000):
            t_ops, train_loss = sess.run([train_step, loss],
                                         feed_dict={inputs: imgs, label_gt: labels, box_gt: control_points})

            if i % 5 == 0:
                print('Epoch: {}, Train Loss: {}'.format(i, train_loss))




if __name__ == '__main__':
    tf.app.run()