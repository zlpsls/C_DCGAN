import config
import numpy as np
import tensorflow as tf
import time
from ops import *
from model import cDCGAN
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test(config, sess):
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        data_list = f.read().splitlines()
    data_len = len(data_list)
    with open(config.DATA_FLIST[config.DATASET][1]) as f:
        label_list = f.read().splitlines()
    model = cDCGAN()
    labels = tf.placeholder(tf.uint8, [1], name='labels')
    image_dims = config.IMG_SHAPES
    images = tf.placeholder(tf.float32, [1] + image_dims, name='real_images')
    D, D_logits, cls, cls_logits = model.discriminator(images)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    show_all_variables()

    log_prefix = config.checkpoint_dir + '/' + '_'.join(
        [config.DATASET, config.MODEL_NAME])
    could_load, checkpoint_counter = load(log_prefix, saver, sess)
    if could_load:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    i = 0
    order = np.random.permutation(data_len)
    for idx in range(100):
        batch_data_files = []
        batch_label = []
        batch_data_files.append(data_list[order[idx]])
        batch_label.append(label_list[order[idx]])
        batch_data = [
            get_image(batch_file, grayscale=True) for batch_file in batch_data_files]
        batch_images = np.array(batch_data).astype(np.float32)
        batch_images = np.expand_dims(batch_images, axis=3)
        batch_label = np.array(batch_label).astype(np.uint8)

        feed_dict = {images: batch_images,
                     labels: batch_label}
        cl = sess.run(tf.argmax(cls, axis=1), feed_dict=feed_dict)
        if cl[0] == batch_label[0]:
            i += 1
        print("Epoch: [%4d], real: %d, d: %d" % (idx, batch_label[0], cl[0]))
    print(i/100)

if __name__ == '__main__':

    config = config.Config('cDCGAN.yml')
    sess = tf.Session()

    test(config, sess)
