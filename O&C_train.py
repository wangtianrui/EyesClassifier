import tensorflow as tf
import os
import numpy as np
import function

import input
import cv2
import OCnet
import OCnet2

NUM_CLASS = 2

IMG_W = 24
IMG_H = 24
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MAX_STEP = 20000
# testCifar10 = 'E:/python_programes/datas/cifar10/cifar-10-batches-bin/'
# TRAIN_PATH = 'F:/Traindata/faceTF/208x208(2).tfrecords'
TRAIN_PATH = 'data/openANDcloseTrain.tfrecords'
TEST_PIC_HOME = "data/openANDcloseTest/"
# train_log_dir = 'E:/python_programes/trainRES/face_wide_res/'
train_log_dir = 'log/'


def train():
    with tf.name_scope('input'):
        # train_image_batch, train_labels_batch = input.read_cifar10(TRAIN_PATH, batch_size=BATCH_SIZE)
        train_image_batch, train_labels_batch = input.read_and_decode_by_tfrecorder(TRAIN_PATH, BATCH_SIZE)
        # test_image_batch, test_labels_batch = input.read_and_decode_by_tfrecorder(TEST_PATH, BATCH_SIZE)
        print(train_image_batch)
        print(train_labels_batch)

        logits = OCnet.inference(train_image_batch, batch_size=BATCH_SIZE, n_classes=NUM_CLASS, name="train")
        # logits = OCnet2.inference(train_image_batch, batch_size=BATCH_SIZE, num_class=NUM_CLASS)

        loss = function.loss(logits=logits, labels=train_labels_batch)
        accuracy_train = function.accuracy(logits=logits, labels=train_labels_batch)

        my_global_step = tf.Variable(0, name='global_step', trainable=True)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss, global_step=my_global_step)
        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):

            if coord.should_stop():
                break

            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy_train])
            # print('***** Step: %d, loss: %.4f *****' % (step, train_loss))
            if (step % 50 == 0) or (step == MAX_STEP - 1):
                print('***** Step: %d, loss: %.4f' % (step, train_loss))
            if (step % 200 == 0) or (step == MAX_STEP - 1):
                print('***** Step: %d, loss: %.4f,train Set accuracy: %.4f%% *****'
                      % (step, train_loss, train_accuracy))
            if step % 2000 == 0 or step == MAX_STEP - 1:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('error')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


# %%


train()
