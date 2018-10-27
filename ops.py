import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import cv2
import numpy as np
import scipy.misc


def show_all_variables():
    #显示模型的所有可训练参数
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load(checkpoint_dir, saver, sess):
    import re
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def save(checkpoint_dir, step, saver, sess):
    model_name = "cDCGAN.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def get_image(image_path, grayscale=False):
    image = imread(image_path, grayscale) / 255 * 2 - 1
    return image


def conv_cond_concat(x, y):
    # 把label的维度从[batchsize, 6]变成 [batchsize, 1, 1, 6]，方便后续拼接
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def imread(path, grayscale=False):
    if grayscale:
        return cv2.imread(path, 0).astype(np.float)
    else:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)


def save_images(image, image_path):
    return scipy.misc.imsave(image_path, image)
