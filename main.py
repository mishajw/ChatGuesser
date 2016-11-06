#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

learning_rate = 0.001
training_iters = 1e4
display_step = 1
batch_size = 3
sequence_length = 5

num_classes = 3
num_hidden = 128
num_chars = 2

input_message = tf.placeholder("float", [batch_size, sequence_length, num_chars], name="input")
output_sender = tf.placeholder("float", [batch_size, num_classes], name="output")

weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))


def message_rnn(x, w, b):
    x = tf.split(1, sequence_length, x)
    x = [tf.squeeze(x_, [1]) for x_ in x]

    lstm_cell = rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True, forget_bias=1.0)
    outputs, state = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = outputs[-1]

    logits = tf.matmul(output, w) + b

    return tf.nn.softmax(logits)


pred = message_rnn(input_message, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, output_sender))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_sender, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < training_iters:
        batch_x = np.array([ \
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], \
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], \
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]])

        batch_y = [[0, 1, 0], [0, 1, 0], [0, 1, 9]]

        sess.run(optimizer, feed_dict={input_message: batch_x, output_sender: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={input_message: batch_x, output_sender: batch_y})
            loss = sess.run(cost, feed_dict={input_message: batch_x, output_sender: batch_y})
            print(sess.run(pred, feed_dict={input_message: batch_x}))

            print("Acc: %.5f, Loss: %.5f" % (acc, loss))
