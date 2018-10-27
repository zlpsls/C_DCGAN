import os, time, pickle, random
import numpy as np
import tensorflow as tf
from ops import *
from glob import glob


class cDCGAN(object):
    def generator(self, z, labels, isTrain=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            # initializer
            # 初始化权重和偏置的方法
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)
            # 将label转为独热码表示
            labels = tf.one_hot(labels, 6, dtype=tf.float32)

            cnum = 64
            y_shape = labels.get_shape().as_list()
            y_layer = tf.reshape(labels, [y_shape[0], 1, 1, y_shape[1]])

            # concat layer1
            cat1 = tf.concat([z, labels], 1)
            fc1 = tf.layers.dense(cat1, 1024, activation=tf.nn.leaky_relu, kernel_initializer=w_init)

            # concat layer2
            cat2 = tf.concat([fc1, labels], 1)
            fc2 = tf.layers.dense(cat2, 4 * 4 * 8 * cnum, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
            g0 = tf.reshape(fc2, [-1, 4, 4, 8 * cnum])
            g0 = conv_cond_concat(g0, y_layer)

            deconv1 = tf.layers.conv2d_transpose(g0, 4 * cnum, [3, 3], strides=(2, 2), padding='same',
                                                 name='deconv1', kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
            g1 = conv_cond_concat(lrelu1, y_layer)

            deconv2 = tf.layers.conv2d_transpose(g1, 2 * cnum, [3, 3], strides=(2, 2), padding='same',
                                                 name='deconv2', kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
            g2 = conv_cond_concat(lrelu2, y_layer)

            deconv3 = tf.layers.conv2d_transpose(g2, cnum, [3, 3], strides=(2, 2), padding='same',
                                                 name='deconv3', kernel_initializer=w_init, bias_initializer=b_init)
            lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
            g3 = conv_cond_concat(lrelu3, y_layer)

            deconv4 = tf.layers.conv2d_transpose(g3, 1, [5, 5], strides=(2, 2), padding='same',
                                                 name='deconv4', kernel_initializer=w_init, bias_initializer=b_init)
            output = tf.nn.tanh(deconv4)

            return output, labels

    def discriminator(self, x, isTrain=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # initializer
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)

            cnum = 64

            conv1 = tf.layers.conv2d(x, cnum, 5, 2, padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, activation=tf.nn.leaky_relu, name='conv1')
            bn1 = tf.layers.batch_normalization(conv1, training=isTrain)
            conv2 = tf.layers.conv2d(bn1, 2 * cnum, 3, 2, padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, activation=tf.nn.leaky_relu, name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=isTrain)
            conv3 = tf.layers.conv2d(bn2, 4 * cnum, 3, 2, padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, activation=tf.nn.leaky_relu, name='conv3')
            bn3 = tf.layers.batch_normalization(conv3, training=isTrain)
            conv4 = tf.layers.conv2d(bn3, 8 * cnum, 3, 2, padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, activation=tf.nn.leaky_relu, name='conv4')

            # output real or false
            fla = tf.layers.flatten(conv4, name='flatten')
            fc1 = tf.layers.dense(fla, 1024, name='fc1')
            output = tf.layers.dense(fc1, 1, name='dout')

            # output cls
            # 共享fc1层
            cls = tf.layers.dense(fc1, 6, name='cls')
            return tf.sigmoid(output), output, tf.nn.softmax(cls), cls
