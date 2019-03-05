import tensorflow as tf
import re
import os

def get_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def get_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=False)

def fc_layer(input_tensor, hidden_size=500):
    w = get_weight([input_tensor.shape.as_list()[1], hidden_size])
    b = get_bias([hidden_size])

    return tf.matmul(input_tensor, w)+b

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

# input and output
x = tf.placeholder(tf.float32, shape=[None, 784], name="input")
y = tf.placeholder(tf.float32, shape=[None, 10], name="output")

# global variables
start = tf.Variable(0, tf.int32, name="start")
total, batchSize = 60000, 100

# network structure
fc1 = fc_layer(x)
fc2 = fc_layer(fc1, 10)
y_ = fc2

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# visualize scalar

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# visualize network

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./draft_fc", sess.graph);

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./draft_fc', sess.graph)
    if os.path.isdir("./checkpoint_fc"):
        saver.restore(sess, "./checkpoint_fc/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(start.eval()+1, int(total/batchSize)):
        x_train, y_train = next_batch(batchSize, i)
        update_start = start.assign(i)
        batch = sess.run([x_train, y_train])
        if i % 10 == 0:
            sess.run(update_start)
            summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch[0], y: batch[1]})
            print("step {}, training accuracy {}".format(int((i+1)/10), train_accuracy))
            train_writer.add_summary(summary, int(i/10))
            saver.save(sess, "./checkpoint_fc/model.ckpt")
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})








