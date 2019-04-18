import tensorflow as tf
import numpy as np
import os
from time import time

from model.factory import model_factory
from dataset.hazy_person import provider
import config
from utils.logging import logger

batch_size = 5
# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'model_name', 'NightFusion',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', 'se_block',
    '''The name of attention module to apply.
    For prioriboxes_mbn, must be "se_block" 
    , "cbam_block" or None; For prioriboxes_vgg, must
    be None
    ''')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_string(
    'summary_dir', './summary/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'batch_size', 5, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'f_log_step', 100,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_summary_step', 10,
    'The frequency with which the model is saved, in step.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 100,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'training_step', None,
    'when training step bigger than training_step, training would stop')

#### config only for prioriboxes_mbn ####
tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'support mobilenet_v1 and mobilenet_v2')

tf.app.flags.DEFINE_boolean(
    'multiscale_feats', True,
    'whether combine different scale features')


FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))
bboxes_gt = tf.placeholder(tf.float32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 4))
label_gt = tf.placeholder(tf.int32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 1))
global_step = tf.Variable(0, trainable=False, name='global_step')


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

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step= global_step)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=session_config) as sess:
        sess.run(init)
        logger.info('TF variables init success...')

        pd = provider(batch_size=batch_size, for_what='train', whether_aug=True)
        avg_det_loss = 0.
        avg_clf_loss = 0.
        avg_time = 0.
        while(True):
            start = time()
            imgs, labels, control_points = pd.load_batch()
            imgs = np.array(imgs)
            labels = np.array(labels)
            control_points = np.array(control_points)
            t_ops, train_loss, current_step, d_loss, c_loss = \
                sess.run([train_step, loss, global_step, det_loss, clf_loss],
                feed_dict={inputs: imgs, label_gt: labels, box_gt: control_points})
            t = round(time() - start, 3)

            if FLAGS.f_log_step != None:
                ## caculate average loss ##
                step = current_step % FLAGS.f_log_step
                avg_det_loss = (avg_det_loss * step + d_loss) / (step + 1.)
                avg_clf_loss = (avg_clf_loss * step + c_loss) / (step + 1.)
                avg_time = (avg_time * step + t) / (step + 1.)
                if current_step%FLAGS.f_log_step == FLAGS.f_log_step-1:
                    ## print info ##
                    logger.info('Step%s det_loss:%s clf_loss:%s time:%s'%(str(current_step),
                                                                            str(avg_det_loss),
                                                                            str(avg_clf_loss),
                                                                            str(avg_time)))
                    avg_det_loss = 0.
                    avg_clf_loss = 0.


            if FLAGS.f_save_step != None:
                if current_step%FLAGS.f_save_step == FLAGS.f_save_step-1:
                    ## save model ##
                    logger.info('Saving model...')
                    model_name = os.path.join(FLAGS.train_dir,FLAGS.model_name+'.model')
                    saver.save(sess, model_name)
                    logger.info('Save model sucess...')




if __name__ == '__main__':
    tf.app.run()