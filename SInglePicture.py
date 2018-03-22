import tensorflow as tf
import numpy as np
import OCnet

test_batch_size = 1
num_class = 2

CHECKPOINT = "F:/Traindata/eyes/result/"  #


def inference(image): # 1 x 24 x 24 x 3
    logits = OCnet.inference(image, test_batch_size, num_class)
    results = tf.arg_max(logits, 1)
    saver = tf.train.Saver()

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
                    result = sess.run(results)
                    print("预测结果是"+result)
                    i += 1
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


