import tensorflow as tf
import function
import input
import OCnet
import matplotlib.pyplot as plt
import numpy as np
import cv2

CHECKPOINT = 'F:/Traindata/eyes/result/'
TEST_PATH = "F:/Traindata/eyes/openANDclosetest.tfrecords"


def plot_images(images, labels):
    for image in images:
        cv2.imshow("test", image)
        cv2.waitKey(0)


image_batch, label_batch = input.read_and_decode_by_tfrecorder(TEST_PATH, 1080, False)
logits = OCnet.inference(image_batch, 1080, 2, "train")
saver = tf.train.Saver()
accuracy = function.accuracy(logits, label_batch)

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT)
    i = 0
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        try:
            while not coord.should_stop() and i < 1:
                # image = sess.run(image_batch)
                # print(image)
                log = sess.run(accuracy)
                # plot_images(image_batch, label_batch)
                print(log)
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
