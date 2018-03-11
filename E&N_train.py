import tensorflow as tf
import input
import function
import ENnet
import os

TEST_PATH = "F:/Traindata/eyes/eyestest.tfrecords"
TRAIN_PATH = "F:/Traindata/eyes/eyes.tfrecords"

RESULT_PATH = "F:/Traindata/eyes/result/"

TRAIN_BATCH_SIZE = 23
TEST_BATCH_SIZE = 2000
LEARNING_RATE = 0.0001
NUME_CLASS = 2
MAX_STEP = 30000


def train():
    image_batch, label_batch = input.read_and_decode_by_tfrecorder(TRAIN_PATH, TRAIN_BATCH_SIZE, True)
    logits = ENnet.inference(image_batch, TRAIN_BATCH_SIZE, NUME_CLASS)
    loss = function.loss(logits, label_batch)
    accuracy = function.accuracy(logits, label_batch)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name="global_step")
    train_op = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)

        try:
            for step in range(MAX_STEP + 1):
                if coord.should_stop():
                    print("coord is stop")
                    break
                acc, _, losses = sess.run([accuracy, train_op, loss])
                if step % 100 == 0:
                    print("step : %d ,loss: %.4f , accuracy on trainSet: %.4f%%" % (step, losses, acc))
                if step % 10000 == 0 or step == MAX_STEP:
                    checkpoint_dir = os.path.join(RESULT_PATH, "model.ckpt")
                    saver.save(sess, checkpoint_dir, global_step=step)
        except tf.errors.OutOfRangeError:
            print("error")
        finally:
            coord.request_stop()
        coord.join(threads=thread)
        sess.close()


def eval():
    image_batch, label_batch = input.read_and_decode_by_tfrecorder(TEST_PATH, TEST_BATCH_SIZE, False)
    logits = ENnet.inference(image_batch, TEST_BATCH_SIZE, NUME_CLASS)
    accuracy = function.accuracy(logits, label_batch)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)
        ckpt = tf.train.get_checkpoint_state(RESULT_PATH)
        i = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            try:
                while not coord.should_stop() and i < 1:
                    acc = sess.run(accuracy)
                    print("%.4f%%" % acc)
                    i += 1
            except tf.errors.OutOfRangeError:
                print("out of range eroor")
            finally:
                coord.request_stop()
            coord.join(threads=thread)
            sess.close()


if __name__ == "__main__":
    eval()