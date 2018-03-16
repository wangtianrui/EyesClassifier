import tensorflow as tf
import function
import input
import OCnet
import matplotlib.pyplot as plt
import numpy as np
import cv2

CHECKPOINT = 'F:/Traindata/eyes/result/'
TEST_PATH = "F:/Traindata/eyes/openANDclosetest.tfrecords"
TEST_BATCH_SIZE = 50


def plot_images(images, labels, results):
    # image = tf.cast(images, dtype=tf.uint8)
    for i in np.arange(0, TEST_BATCH_SIZE):
        e = i + 1
        plt.subplot(8, 7, i + 1)
        plt.axis('off')
        title = "L:" + str(labels[i]) + " R:" + str(results[i])
        plt.title(title, fontsize=14)
        plt.subplots_adjust(hspace=0.5)
        plt.imshow(images[i])
    plt.show()


image_batch, label_batch, image_raw = input.read_and_decode_by_tfrecorder_eye(TEST_PATH, TEST_BATCH_SIZE, False)
print(image_raw)
logits = OCnet.inference(image_batch, TEST_BATCH_SIZE, 2, "train")
labels = tf.argmax(label_batch, 1)
results = tf.argmax(logits, 1)
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
                image, label, result = sess.run([image_raw, labels, results])
                log = sess.run(accuracy)
                # plot_images(image_batch, label_batch)
                print(log)
                plot_images(image, label, result)

                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
