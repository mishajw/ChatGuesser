#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from datetime import datetime

import data
from model import ChatGuesserModel

max_sequence_length = 50
batch_size = 64
training_percentage = 0.75
training_iters = 1e4
display_step = 10

model = ChatGuesserModel(max_sequence_length)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    inputs, outputs, test_inputs, test_outputs, name_set = data.get_data(training_percentage, max_sequence_length)

    now = datetime.now()
    summary_train_writer = tf.train.SummaryWriter(
        "/tmp/tb_chat_guesser/" + now.strftime("%Y%m%d-%H%M%S") + "/train",
        sess.graph)
    summary_test_writer = tf.train.SummaryWriter(
        "/tmp/tb_chat_guesser/" + now.strftime("%Y%m%d-%H%M%S") + "/test",
        sess.graph)

    while step < training_iters:
        batch_step = step % int(len(outputs) / batch_size)

        batch_x = np.array(
            inputs[batch_step * batch_size:(batch_step + 1) * batch_size]
        )

        batch_y = np.array(
            outputs[batch_step * batch_size:(batch_step + 1) * batch_size]
        )

        _, train_summaries = sess.run(
            [model.optimizer, model.all_summaries],
            feed_dict={model.messages: batch_x, model.senders: batch_y})

        summary_train_writer.add_summary(train_summaries, step)

        if step % display_step == 0:
            acc, loss, test_summaries = sess.run(
                [model.accuracy, model.cost, model.all_summaries],
                feed_dict={model.messages: inputs, model.senders: outputs})

            print("Acc: %.5f, Loss: %.5f" % (acc, loss))
            summary_test_writer.add_summary(test_summaries, step)

        step += 1
