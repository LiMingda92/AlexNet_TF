#coding=utf-8

import tensorflow as tf
import numpy as np
from datetime import datetime
from alexnet import *

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    #print 'label的样子是：', label
    return img, label

def train():

    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 17], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = alexnet(x, keep_prob)

    loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    pred = tf.argmax(output, 1)
    truth = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
    saver = tf.train.Saver()

    img, label = read_and_decode("./train.tfrecords")
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=32, capacity=2000,
                                                min_after_dequeue=1000)
    labels = tf.one_hot(label_batch,17,1,0)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2500):
            #print '未转换的label样子是： ', sess.run(label)
            batch_xs, batch_ys = sess.run([img_batch, labels])
            #print '转换为onehot的label样子是： ', batch_ys, len(batch_ys), '个'
            _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})
            if i%10 == 0:
                train_arr = accuracy.eval(feed_dict={x:batch_xs, y: batch_ys, keep_prob: 1.0})
                print "%s: Step [%d]  Loss : %f, training accuracy :  %g" % (datetime.now(), i, loss_val, train_arr)

        coord.request_stop()
        coord.join()
        saver.save(sess,'./model/model.ckpt')
        #saver.restore(sess, './model.ckpt')读取参数
        #tf.train.write_graph(sess.graph_def,'.','pbtxt')
def main(argv=None):
    train()
if __name__ == '__main__':
    tf.app.run()


