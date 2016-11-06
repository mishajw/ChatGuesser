#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

import data


class ChatGuesserModel:
    def __init__(self, max_sequence_length):
        self.learning_rate = 0.0001
        self.max_sequence_length = max_sequence_length

        self.num_classes = 18
        self.num_hidden = 64
        self.num_chars = 128

        self.messages = tf.placeholder("float", [None, self.max_sequence_length], name="input")
        self.senders = tf.placeholder("float", [None], name="output")

        self.weights = tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        self.biases = tf.Variable(tf.random_normal([self.num_classes]))

        self.optimizer = None
        self.accuracy = None
        self.cost = None

        self.model()

    def model(self):
        pred = self.message_rnn(self.messages, self.weights, self.biases)

        output_one_hot = tf.one_hot(tf.cast(self.senders, tf.int32), 18, axis=1)
        softmax = tf.nn.softmax_cross_entropy_with_logits(pred, output_one_hot)
        self.cost = tf.reduce_mean(softmax)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def message_rnn(self, x, w, b):
        x = tf.one_hot(tf.cast(x, tf.int32), self.num_chars, axis=2)

        lstm_cell = rnn_cell.BasicLSTMCell(self.num_hidden, state_is_tuple=True, forget_bias=1.0)
        outputs, state = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=self.real_length(x))
        output = self.get_last_output(outputs)

        logits = tf.matmul(output, w) + b

        return tf.nn.softmax(logits)

    def get_last_output(self, outputs):
        last_index = tf.shape(outputs)[1] - 1
        outputs_rs = tf.transpose(outputs, [1, 0, 2])

        # TODO: find out what this does
        return tf.nn.embedding_lookup(outputs_rs, last_index)

    def real_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


max_sequence_length = 50
batch_size = 64
training_percentage = 0.9
training_iters = 1e4
display_step = 50

model = ChatGuesserModel(max_sequence_length)
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

        sess.run(model.optimizer, feed_dict={model.messages: batch_x, model.senders: batch_y})

        if step % display_step == 0:
            acc, loss = sess.run([model.accuracy, model.cost], feed_dict={model.messages: test_inputs, model.senders: test_outputs})

            print("Acc: %.5f, Loss: %.5f" % (acc, loss))

        step += 1
