import tensorflow as tf
import numpy as np
import os

IMG_W = 24
IMG_H = 24


def read_and_decode_by_tfrecorder_eye(tfrecords_file, batch_size, shuffle=True):
    # 生成队列。该函数后面的参数是一个文件名数组。（可以存放多个文件名，该方法就是将多个文件名进行一个队列化）
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image_raw = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image_raw = tf.reshape(image_raw, [24, 24, 3])
    label = tf.cast(img_features['label'], tf.int64)
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image_raw, label],
            batch_size=batch_size,
            num_threads=8,
            capacity=50,
            min_after_dequeue=30)
    else:
        image_raw, label_batch = tf.train.batch(
            [image_raw, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=500)

    n_classes = 2
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    image_batch = tf.cast(image_raw, tf.float32)

    return image_batch, label_batch , image_raw
    #, image_raw


def read_and_decode_by_tfrecorder(tfrecords_file, batch_size, shuffle=True):
    # 生成队列。该函数后面的参数是一个文件名数组。（可以存放多个文件名，该方法就是将多个文件名进行一个队列化）
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image_raw = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image_raw = tf.reshape(image_raw, [24, 24, 3])
    image_raw = tf.cast(image_raw, tf.float32)
    label = tf.cast(img_features['label'], tf.int64)
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image_raw, label],
            batch_size=batch_size,
            num_threads=8,
            capacity=50,
            min_after_dequeue=30)
    else:
        image_batch, label_batch = tf.train.batch(
            [image_raw, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=500)

    n_classes = 2
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch
    #, image_raw


def read_cifar10(data_dir, batch_size, shuffle=False, is_train=False):
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width * img_height * img_depth

    with tf.name_scope('input'):

        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % ii)
                         for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

        key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        image = tf.image.per_image_standardization(image)  # substract off the mean and divide by the variance

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=64,
                capacity=20000,
                min_after_dequeue=3000)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=64,
                capacity=2000)
        ## ONE-HOT
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch
