#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

import data

learning_rate = 0.0001
training_percentage = 0.75
training_iters = 1e4
display_step = 10
batch_size = 256
max_sequence_length = 5

num_classes = 18
num_hidden = 64
num_chars = 128

input_message = tf.placeholder("float", [None, max_sequence_length], name="input")
output_sender = tf.placeholder("float", [None], name="output")

weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))


def message_rnn(x, w, b):
    x = tf.one_hot(tf.cast(x, tf.int32), num_chars, axis=2)

    lstm_cell = rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True, forget_bias=1.0)
    outputs, state = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=real_length(x))
    output = get_last_output(outputs)

    logits = tf.matmul(output, w) + b

    return tf.nn.softmax(logits)


def get_last_output(outputs):
    last_index = tf.shape(outputs)[1] - 1
    outputs_rs = tf.transpose(outputs, [1, 0, 2])

    # TODO: find out what this does
    return tf.nn.embedding_lookup(outputs_rs, last_index)


def real_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


pred = message_rnn(input_message, weights, biases)

output_one_hot = tf.one_hot(tf.cast(output_sender, tf.int32), 18, axis=1)
softmax = tf.nn.softmax_cross_entropy_with_logits(pred, output_one_hot)
cost = tf.reduce_mean(softmax)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    inputs, outputs, test_inputs, test_outputs, name_set = data.get_data(training_percentage, max_sequence_length)

    while step < training_iters:
        batch_x = np.array(
            inputs[step * batch_size:(step + 1) * batch_size]
        )

        batch_y = np.array(
            outputs[step * batch_size:(step + 1) * batch_size]
        )

        if len(batch_x) != batch_size:
            continue

        sess.run(optimizer, feed_dict={input_message: batch_x, output_sender: batch_y})

        if step % display_step == 0:
            acc, loss = sess.run([accuracy, cost], feed_dict={input_message: test_inputs, output_sender: test_outputs})

            print("Acc: %.5f, Loss: %.5f" % (acc, loss))

        step += 1
