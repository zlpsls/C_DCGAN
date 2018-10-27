import config
import numpy as np
import tensorflow as tf
import time
from ops import *
from model import cDCGAN
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(config, sess):
    # training data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        data_list = f.read().splitlines()
    data_len = len(data_list)
    with open(config.DATA_FLIST[config.DATASET][1]) as f:
        label_list = f.read().splitlines()

    model = cDCGAN()

    labels = tf.placeholder(tf.uint8, [config.batchsize], name='labels')
    image_dims = config.IMG_SHAPES
    images = tf.placeholder(tf.float32, [config.batchsize] + image_dims, name='real_images')
    z = tf.placeholder(tf.float32, [config.batchsize, config.z_dim], name='z')

    G, hot_label = model.generator(z, labels)
    D, D_logits, cls, cls_logits = model.discriminator(images)
    D_, D_logits_, cls_, cls_logits_ = model.discriminator(G, reuse=True)
    #loss
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
    d_cls_loss = config.log_alpha * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=hot_label, logits=cls_logits))
    gim_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))
    g_cls_loss = config.log_alpha * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=hot_label, logits=cls_logits_))

    #tensorboard显示loss曲线
    tf.summary.scalar("d_loss_real", d_loss_real)
    tf.summary.scalar("d_loss_fake", d_loss_fake)
    tf.summary.scalar("d_cls_loss", d_cls_loss)
    tf.summary.scalar("gim_loss", gim_loss)
    tf.summary.scalar("g_cls_loss", g_cls_loss)

    d_loss = d_loss_real + d_loss_fake + d_cls_loss
    g_loss = gim_loss + g_cls_loss

    tf.summary.scalar("g_loss", g_loss)
    tf.summary.scalar("d_loss", d_loss)

    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    #tensorboard显示当前生成图片
    viz_img = G
    tf.summary.image('generate_img', viz_img, config.VIZ_MAX_OUT)

    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(config.LEARNING_RATE))

    d_optimizer = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    show_all_variables()
    # log dir
    summary_op = tf.summary.merge_all()

    writer = tf.summary.FileWriter("./logs/" + '_'.join(
        [config.DATASET, config.MODEL_NAME]), sess.graph)

    log_prefix = config.checkpoint_dir + '/' + '_'.join(
        [config.DATASET, config.MODEL_NAME])
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = load(log_prefix, saver, sess)
    if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for epoch in range(config.EPOCH):
        # 训练集打乱顺序随机输入
        order = np.random.permutation(data_len)
        batch_idxs = int(data_len // config.batchsize)
        for idx in range(0, batch_idxs):
            order_batch = order[idx * config.batchsize:(idx + 1) * config.batchsize]
            batch_data_files = []
            batch_label = []
            batch_z = np.random.uniform(-1, 1, [config.batchsize, config.z_dim]).astype(np.float32)
            for k in range(config.batchsize):
                batch_data_files.append(data_list[order_batch[k]])
                batch_label.append(label_list[order_batch[k]])
            batch_data = [
                get_image(batch_file, grayscale=True) for batch_file in batch_data_files]
            batch_images = np.array(batch_data).astype(np.float32)
            batch_images = np.expand_dims(batch_images, axis=3)
            batch_label = np.array(batch_label).astype(np.uint8)

            feed_dict = {images: batch_images,
                         labels: batch_label,
                         z: batch_z}

            # Update network
            sess.run(d_optimizer, feed_dict=feed_dict)
            sess.run(g_optimizer, feed_dict=feed_dict)

            if np.mod(counter, 20) == 1:
                #每20次mini_batch存一次模型，可以自行修改
                save(log_prefix, counter, saver, sess)
                loss_g, loss_d = sess.run([g_loss, d_loss], feed_dict=feed_dict)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, d_loss: %.8f" \
                      % (epoch, idx, batch_idxs, (time.time() - start_time), loss_g, loss_d))
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            writer.add_summary(summary_str, counter)
            counter += 1


if __name__ == '__main__':

    config = config.Config('cDCGAN.yml')
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    sess = tf.Session()

    # train
    train(config, sess)
