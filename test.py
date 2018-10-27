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
    model = cDCGAN()
    labels = tf.placeholder(tf.uint8, [1], name='labels')
    z = tf.placeholder(tf.float32, [1, config.z_dim], name='z')
    G, hot_label = model.generator(z, labels,)
    D_, D_logits_, cls_, cls_logits_ = model.discriminator(G)

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
    for idx in range(100):
        batch_label = [np.random.randint(0, 6)]
        batch_z = np.random.uniform(-1, 1, [1, config.z_dim]).astype(np.float32)
        batch_label = np.array(batch_label).astype(np.uint8)

        feed_dict = {labels: batch_label,
                     z: batch_z}
        cls, img = sess.run([tf.argmax(cls_, axis=1), G], feed_dict=feed_dict)
        img = np.uint8((np.squeeze(img) + 1) * 255 / 2)
        if cls[0] == batch_label[0]:
            i += 1
        save_images(img, './%s/%d_%d_%d.jpg' % (config.test_dir, idx, batch_label[0], cls[0]))
        print("Epoch: [%4d], real: %d, d: %d" % (idx, batch_label[0], cls[0]))
    print(i/100)

if __name__ == '__main__':

    config = config.Config('cDCGAN.yml')
    if not os.path.exists(config.test_dir):
        os.makedirs(config.test_dir)
    sess = tf.Session()

    test(config, sess)
