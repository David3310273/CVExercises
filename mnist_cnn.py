import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer
# from tensorflow.python.tools import inspect_checkpoint as chkp
import re
import os.path

def loadData(path=""):
    pass;

"""

Lenet5神经网络复现

写神经网络代码重点在于：

1. 搞清每一步的数据维度和意义
2. 注意重点展示的是单步的计算方法，而不是处理数据的过程，是面向函数的编程范式

"""

def next_batch(batchSize=100, epoch=1):
    template = tf.gfile.Glob("PATH_EXPRESSION");
    x_train, y_train = [], []

    for path in template[(epoch-1)*batchSize:epoch*batchSize]:
        info = re.findall(r"PATH_EXPRESSION", path)[0]
        number = int(info.split("_")[1])
        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image = tf.reshape(image_decoded, [784, ])
        x_train.append(image)
        y_train.append(tf.one_hot(number, depth=10))
    return x_train, y_train

# build cnn structure
def buildModel():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    y_predict = y_conv

    # y_predict = tf.Print(y_conv, [y_conv, tf.shape(y_conv)], "the result is: ", summarize=50)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    # y_predict = tf.Print(y_conv, [y_conv], "this is result: ", summarize=10)
    start = tf.Variable(0, name="start", dtype=tf.int32)
    saver = tf.train.Saver()

    merged = tf.summary.merge_all()

    # draw graph on tensorboard
    # with tf.Session() as sess:
    #     tf.summary.FileWriter("./draft", sess.graph);

    # return bool type tensor
    # tf.argmax: return index of maximun value in a tenso

    batchSize = 100
    # chkp.print_tensors_in_checkpoint_file("./checkpoint/model.ckpt", tensor_name='start', all_tensors=False)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./draft', sess.graph)
        if os.path.isdir('./checkpoint'):
            saver.restore(sess, "./checkpoint/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(start.eval()+1, int(60000/batchSize)):
            x_train, y_train = next_batch(batchSize, i+1)
            update_start = start.assign(i)
            batch = sess.run([x_train, y_train])
            if (i+1) % 10 == 0:
                sess.run(update_start)
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch[0],y_: batch[1],keep_prob: 1.0})
                print("step {}, training accuracy {}".format(int((i+1)/10), train_accuracy))
                train_writer.add_summary(summary, i)
                saver.save(sess, "./checkpoint/model.ckpt")
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

if __name__ == '__main__':
    # print(help(os.path))
    buildModel()
    # b1 = tf.Variable(tf.constant(0.1, shape=[3]));
