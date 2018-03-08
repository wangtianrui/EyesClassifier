import tensorflow as tf
import os
import numpy as np
import function

import input
import cv2
import net

NUM_CLASS = 2
# IMG_W = 208
# IMG_H = 208
IMG_W = 24
IMG_H = 24
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_STEP = 50000
# testCifar10 = 'E:/python_programes/datas/cifar10/cifar-10-batches-bin/'
# TRAIN_PATH = 'F:/Traindata/faceTF/208x208(2).tfrecords'
TRAIN_PATH = 'F:/Traindata/eyes/eyes.tfrecords'
TEST_PATH = 'F:/Traindata/eyes/openANDclosetest.tfrecords'
TEST_PIC_HOME = "F:/Traindata/eyes/openANDcloseTest/"
# train_log_dir = 'E:/python_programes/trainRES/face_wide_res/'
train_log_dir = 'F:/Traindata/eyes/result/'


def train():
    with tf.name_scope('input'):
        # train_image_batch, train_labels_batch = input.read_cifar10(TRAIN_PATH, batch_size=BATCH_SIZE)
        train_image_batch, train_labels_batch = input.read_and_decode_by_tfrecorder(TRAIN_PATH, BATCH_SIZE)
        # test_image_batch, test_labels_batch = input.read_and_decode_by_tfrecorder(TEST_PATH, BATCH_SIZE)
        print(train_image_batch)
        print(train_labels_batch)

        logits = net.inference(train_image_batch, batch_size=BATCH_SIZE, n_classes=NUM_CLASS, name="train")

        print(logits)
        loss = function.loss(logits=logits, labels=train_labels_batch)
        # accuracy_logits = net.inference(test_image_batch, batch_size=BATCH_SIZE, n_classes=NUM_CLASS, name="test")
        # accuracy_test = function.accuracy(logits=accuracy_logits, labels=test_labels_batch)
        accuracy_train = function.accuracy(logits=logits, labels=train_labels_batch)

        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        # train_op = function.optimize(loss=loss, learning_rate=LEARNING_RATE, global_step=my_global_step)
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
            if (step % 50 == 0) or (step == MAX_STEP):
                print('***** Step: %d, loss: %.4f' % (step, train_loss))
            if (step % 200 == 0) or (step == MAX_STEP):
                print('***** Step: %d, loss: %.4f,train Set accuracy: %.4f%% *****'
                      % (step, train_loss, train_accuracy))
            if step % 2000 == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('error')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()



# %%

def load_image():
    all_classes = []
    all_images = []
    all_labels = []

    for i in os.listdir(TEST_PIC_HOME):
        curren_dir = os.path.join(TEST_PIC_HOME, i)
        if os.path.isdir(curren_dir):
            all_classes.append(i)
            for img in os.listdir(curren_dir):
                if img.endswith('png') or img.endswith('bmp') or img.endswith('jpg'):
                    all_images.append(os.path.join(curren_dir, img))
                    if (i == 'close'):
                        all_labels.append(0)
                    if (i == 'open'):
                        all_labels.append(1)
        else:
            print(curren_dir, " doesnt exist")

    return all_classes, all_images, all_labels

def input_map_fn(image_path, label):
    one_hot = tf.one_hot(label, NUM_CLASS)
    image_f = tf.read_file(image_path)
    image_decode = tf.image.decode_image(image_f, channels=3)
    image_decode = tf.cast(image_decode,dtype=tf.float32)
    return image_decode, one_hot


train()
