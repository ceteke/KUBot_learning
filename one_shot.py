import tensorflow as tf
from data_handler import DataHandler

hidden_size = 200
input_size = 51
output_size = 3

x_t = tf.placeholder(tf.float32, [1, input_size], name='x_t')
y_t = tf.placeholder(tf.float32, [1, output_size], name='y_t')

with tf.name_scope("lstm"):
    state = tf.Variable(tf.zeros([2, 200]), tf.float32)

    W_z = tf.Variable(tf.random_normal([input_size, hidden_size]), name="W_z")
    W_i = tf.Variable(tf.random_normal([input_size, hidden_size]), name="W_i")
    W_f = tf.Variable(tf.random_normal([input_size, hidden_size]), name="W_f")
    W_o = tf.Variable(tf.random_normal([input_size, hidden_size]), name="W_o")

    R_z = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name="R_z")
    R_i = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name="R_i")
    R_f = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name="R_f")
    R_o = tf.Variable(tf.random_normal([hidden_size, hidden_size]), name="R_o")

    b_z = tf.Variable(tf.random_normal([1, hidden_size]), name="b_z")
    b_i = tf.Variable(tf.random_normal([1, hidden_size]), name="b_i")
    b_f = tf.Variable(tf.random_normal([1, hidden_size]), name="b_f")
    b_o = tf.Variable(tf.random_normal([1, hidden_size]), name="b_o")

    p_i = tf.Variable(tf.random_normal([hidden_size]), name="p_i")
    p_f = tf.Variable(tf.random_normal([hidden_size]), name="p_f")
    p_o = tf.Variable(tf.random_normal([hidden_size]), name="p_o")

with tf.name_scope("dense"):
    hidden_layer = tf.Variable(tf.random_normal([hidden_size, output_size], name="W_r", dtype=tf.float32))
    b_r = tf.Variable(tf.random_normal([1, output_size], name="b_r", dtype=tf.float32))


with tf.name_scope("lstm_step"):
    g = h = tf.tanh
    y = tf.reshape(state[0], [1, hidden_size])
    c = tf.reshape(state[1], [1, hidden_size])

    z = g(tf.matmul(x_t, W_z) + tf.matmul(y, R_z) + b_z)
    i = tf.sigmoid(tf.matmul(x_t, W_i) + tf.matmul(y, R_i) + tf.multiply(c, p_i) + b_i)
    f = tf.sigmoid(tf.matmul(x_t, W_f) + tf.matmul(y, R_f) + tf.multiply(c, p_f) + b_f)
    c = tf.multiply(i, z) + tf.multiply(f, c)
    o = tf.sigmoid(tf.matmul(x_t, W_o) + tf.matmul(y, R_o) + tf.multiply(c, p_o) + b_o)
    y = tf.multiply(h(c), o)

    tf.summary.histogram('W_f', W_f)
    tf.summary.histogram('p_f', R_f)
    tf.summary.histogram('b_f', b_f)

    tf.summary.histogram('W_i', W_i)
    tf.summary.histogram('R_i', R_i)
    tf.summary.histogram('b_i', b_i)

    tf.summary.histogram('W_o', W_o)
    tf.summary.histogram('R_o', R_o)
    tf.summary.histogram('b_o', b_o)

    tf.summary.histogram('W_z', W_z)
    tf.summary.histogram('R_z', R_z)
    tf.summary.histogram('b_z', b_z)

    state = tf.concat([y, c], 0)


with tf.name_scope("dense_feedforward"):
    y_pred = tf.tanh(tf.matmul(y, hidden_layer) + b_r)
    tf.summary.histogram('hidden_layer', hidden_layer)
    tf.summary.histogram('b_r', b_r)

with tf.name_scope("train"):
    cost = tf.reduce_mean(tf.square(y_pred - y_t))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    tf.summary.scalar('loss', cost)

dh = DataHandler()
dh.collect_data()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('board/train/14', sess.graph)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    sess.run(init)

    for a in dh.actions:
        a.split_train_test(0.05)
        a.scale_dataset()
        print "Training.."
        for i in range(len(a.X_train)):
            x_train = a.X_train[i].reshape(1, 51)
            y_train = a.y_train_p[i].reshape(1, 3)
            summary, _ = sess.run([merged, optimizer], {x_t: x_train, y_t: y_train})
            train_writer.add_summary(summary, i)


